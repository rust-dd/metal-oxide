use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
    time::Duration,
};

use super::{
    failure::CommandFailure,
    sync::{Condvar, Mutex, MutexGuard, lock},
};
use crate::{Error, Result};

#[derive(Default)]
struct State {
    result: Option<std::result::Result<SubmissionReport, CommandFailure>>,
    waker: Option<Waker>,
}

/// The single result of a committed batch, written once by its native callback.
#[derive(Default)]
pub(super) struct Completion {
    state: Mutex<State>,
    ready: Condvar,
}

impl Completion {
    pub(super) fn finish(&self, result: std::result::Result<SubmissionReport, CommandFailure>) {
        let waker = {
            let mut state = lock(&self.state);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            state.waker.take()
        };
        self.ready.notify_all();
        if let Some(waker) = waker {
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

    pub(super) fn wait(&self) -> Result<SubmissionReport> {
        self.finished().result.clone().unwrap().map_err(Error::from)
    }

    pub(super) fn poll(&self, cx: &mut Context<'_>) -> Poll<Result<SubmissionReport>> {
        let mut state = lock(&self.state);
        match &state.result {
            Some(result) => Poll::Ready(result.clone().map_err(Error::from)),
            None => {
                if !state
                    .waker
                    .as_ref()
                    .is_some_and(|w| w.will_wake(cx.waker()))
                {
                    state.waker = Some(cx.waker().clone());
                }
                Poll::Pending
            }
        }
    }
}

/// Measurements supplied by native command completion feedback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubmissionReport {
    /// GPU execution time for the entire batch; None for empty batches or unavailable timestamps.
    pub gpu_duration: Option<Duration>,
}

impl SubmissionReport {
    pub(super) fn from_gpu_times(start: f64, end: f64) -> Self {
        let gpu_duration = (start.is_finite() && start > 0.0 && end.is_finite() && end >= start)
            .then(|| Duration::try_from_secs_f64(end - start).ok())
            .flatten();
        Self { gpu_duration }
    }
}

/// A committed GPU batch. Dropping it leaves execution and resource retention active.
///
/// Await it or call `wait` to receive the command's completion status. CPU buffer
/// access synchronizes pending GPU work even if the submission was dropped or forgotten.
#[must_use = "await the submission or call wait to check GPU completion"]
pub struct Submission<'a> {
    pub(super) completion: Arc<Completion>,
    pub(super) marker: PhantomData<&'a ()>,
}

impl Submission<'_> {
    pub fn wait(self) -> Result<SubmissionReport> {
        self.completion.wait()
    }
}

impl Future for Submission<'_> {
    type Output = Result<SubmissionReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.completion.poll(cx)
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use objc2_foundation::{NSError, NSString};
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        task::Wake,
    };

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
        assert_eq!(
            SubmissionReport::from_gpu_times(12.0, 12.25).gpu_duration,
            Some(std::time::Duration::from_millis(250))
        );
        for (start, end) in [
            (0.0, 0.0),
            (2.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (1.0, f64::INFINITY),
        ] {
            assert_eq!(
                SubmissionReport::from_gpu_times(start, end).gpu_duration,
                None
            );
        }
    }

    #[test]
    fn completion_wakes_the_latest_waiter_and_preserves_errors() {
        let completion = Completion::default();
        let first = Arc::new(Counter(AtomicUsize::new(0)));
        let second = Arc::new(Counter(AtomicUsize::new(0)));
        for counter in [&first, &second] {
            let waker = Waker::from(Arc::clone(counter));
            assert!(
                completion
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending()
            );
        }
        completion.finish(Err(CommandFailure::Other("test command failure".into())));
        assert_eq!(first.0.load(Ordering::Relaxed), 0);
        assert_eq!(second.0.load(Ordering::Relaxed), 1);
        assert_eq!(
            completion.wait().unwrap_err().to_string(),
            "Metal command failed: test command failure"
        );
        assert!(
            completion
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_ready()
        );
    }
}
