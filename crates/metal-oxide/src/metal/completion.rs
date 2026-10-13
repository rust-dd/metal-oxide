use std::{
    task::{Context, Poll, Waker},
    time::Duration,
};

use super::{
    failure::CommandFailure,
    sync::{Condvar, Mutex, MutexGuard, lock},
};
use crate::{Error, Result};

/// The GPU execution time of a command buffer, when its timestamps are available.
pub(super) type GpuTime = Option<Duration>;

#[derive(Default)]
struct State {
    result: Option<std::result::Result<GpuTime, CommandFailure>>,
    wakers: Vec<Waker>,
}

/// The single result of a committed command buffer, written once by its native callback.
#[derive(Default)]
pub(super) struct Completion {
    state: Mutex<State>,
    ready: Condvar,
}

impl Completion {
    pub(super) fn finish(&self, result: std::result::Result<GpuTime, CommandFailure>) {
        let wakers = {
            let mut state = lock(&self.state);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            std::mem::take(&mut state.wakers)
        };
        self.ready.notify_all();
        for waker in wakers {
            // A user waker must not unwind through the native completion callback.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }

    pub(super) fn is_finished(&self) -> bool {
        lock(&self.state).result.is_some()
    }

    /// Blocks until the native callback has published the result.
    pub(super) fn block(&self) {
        drop(self.finished());
    }

    fn finished(&self) -> MutexGuard<'_, State> {
        let mut state = lock(&self.state);
        while state.result.is_none() {
            state = self
                .ready
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
        state
    }

    pub(super) fn wait(&self) -> Result<GpuTime> {
        self.finished().result.clone().unwrap().map_err(Error::from)
    }

    pub(super) fn poll(&self, cx: &mut Context<'_>) -> Poll<Result<GpuTime>> {
        let mut state = lock(&self.state);
        match &state.result {
            Some(result) => Poll::Ready(result.clone().map_err(Error::from)),
            None => {
                register(&mut state.wakers, cx.waker());
                Poll::Pending
            }
        }
    }

    /// Whether the command buffer has finished; otherwise wakes `cx` when it does.
    pub(super) fn poll_finished(&self, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = lock(&self.state);
        if state.result.is_some() {
            return Poll::Ready(());
        }
        register(&mut state.wakers, cx.waker());
        Poll::Pending
    }
}

/// Keeps one entry per distinct waker; a re-polled future replaces none of the others.
fn register(wakers: &mut Vec<Waker>, waker: &Waker) {
    if !wakers.iter().any(|existing| existing.will_wake(waker)) {
        wakers.push(waker.clone());
    }
}

/// Converts native GPU timestamps in seconds into a duration, rejecting missing or invalid ones.
pub(super) fn gpu_time(start: f64, end: f64) -> GpuTime {
    (start.is_finite() && start > 0.0 && end.is_finite() && end >= start)
        .then(|| Duration::try_from_secs_f64(end - start).ok())
        .flatten()
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use objc2_foundation::{NSError, NSString};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::task::Wake;

    struct Counter(AtomicUsize);
    impl Wake for Counter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore = "creates an NSError through Objective-C")]
    fn terminal_memory_errors_survive_wait_and_poll() {
        for (domain, code, memory) in [
            ("MTLCommandBufferErrorDomain", 8, true),
            ("MTL4CommandQueueErrorDomain", 3, true),
            ("MTLCommandBufferErrorDomain", 3, false),
            ("MTL4CommandQueueErrorDomain", 8, false),
            ("unrelated", 8, false),
            ("unrelated", 3, false),
        ] {
            // SAFETY: no user-info dictionary or untyped objects are supplied.
            let error = unsafe {
                NSError::errorWithDomain_code_userInfo(&NSString::from_str(domain), code, None)
            };
            let completion = Completion::default();
            completion.finish(Err(CommandFailure::from_native(&error)));
            for _ in 0..2 {
                let error = completion.wait().unwrap_err();
                assert_eq!(matches!(error, Error::OutOfMemory(_)), memory, "{error}");
                assert!(error.to_string().contains(domain));
                let Poll::Ready(Err(error)) =
                    completion.poll(&mut Context::from_waker(Waker::noop()))
                else {
                    panic!("terminal failure was lost");
                };
                assert_eq!(matches!(error, Error::OutOfMemory(_)), memory, "{error}");
            }
        }
    }

    #[test]
    fn native_timestamps_reject_unavailable_and_invalid_measurements() {
        assert_eq!(gpu_time(12.0, 12.25), Some(Duration::from_millis(250)));
        for (start, end) in [
            (0.0, 0.0),
            (2.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert_eq!(gpu_time(start, end), None);
        }
    }

    #[test]
    fn completion_wakes_every_distinct_waiter_once_and_preserves_errors() {
        let completion = Completion::default();
        let first = Arc::new(Counter(AtomicUsize::new(0)));
        let second = Arc::new(Counter(AtomicUsize::new(0)));
        for counter in [&first, &second, &first] {
            let waker = Waker::from(Arc::clone(counter));
            assert!(
                completion
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending()
            );
        }
        let third = Arc::new(Counter(AtomicUsize::new(0)));
        let waker = Waker::from(Arc::clone(&third));
        assert!(
            completion
                .poll_finished(&mut Context::from_waker(&waker))
                .is_pending()
        );
        completion.finish(Err(CommandFailure::Other("test command failure".into())));
        for counter in [&first, &second, &third] {
            assert_eq!(counter.0.load(Ordering::Relaxed), 1);
        }
        assert_eq!(
            completion.wait().unwrap_err().to_string(),
            "Metal command failed: test command failure"
        );
        assert!(
            completion
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_ready()
        );
        assert!(
            completion
                .poll_finished(&mut Context::from_waker(Waker::noop()))
                .is_ready()
        );
    }
}
