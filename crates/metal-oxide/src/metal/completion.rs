use std::{
    cell::RefCell,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    rc::Rc,
    sync::{Arc, Condvar, Mutex},
    task::{Context, Poll, Waker},
};

use crate::{Error, Result};

#[derive(Default)]
struct State {
    result: Option<std::result::Result<(), String>>,
    waker: Option<Waker>,
}

#[derive(Default)]
pub(super) struct Completion {
    state: Mutex<State>,
    ready: Condvar,
}

impl Completion {
    pub(super) fn finish(&self, result: std::result::Result<(), String>) {
        let waker = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
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

    fn is_finished(&self) -> bool {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .result
            .is_some()
    }

    fn wait(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while state.result.is_none() {
            state = self.ready.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        state
            .result
            .as_ref()
            .unwrap()
            .clone()
            .map_err(Error::Command)
    }

    fn poll(&self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match &state.result {
            Some(result) => Poll::Ready(result.clone().map_err(Error::Command)),
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

#[derive(Default)]
pub(super) struct AccessState(RefCell<Vec<Arc<Completion>>>);

impl AccessState {
    pub(super) fn register(&self, completion: &Arc<Completion>) {
        let mut pending = self.0.borrow_mut();
        pending.retain(|state| !state.is_finished());
        pending.push(Arc::clone(completion));
    }

    pub(super) fn synchronize(&self) {
        for completion in self.0.borrow_mut().drain(..) {
            // Written scalars remain valid on command failure; submission reports its error.
            let _ = completion.wait();
        }
    }
}

/// A committed GPU batch. Dropping it leaves execution and resource retention active.
///
/// Await it or call `wait` to receive the command's completion status. CPU buffer
/// access synchronizes pending GPU work even if the submission was dropped or forgotten.
#[must_use = "await the submission or call wait to check GPU completion"]
pub struct Submission<'a> {
    pub(super) completion: Arc<Completion>,
    pub(super) marker: PhantomData<&'a mut Rc<()>>,
}

impl Submission<'_> {
    pub fn wait(self) -> Result<()> {
        self.completion.wait()
    }
}

impl Future for Submission<'_> {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.completion.poll(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        completion.finish(Err("test command failure".into()));
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

    #[test]
    fn buffer_access_waits_for_every_pending_submission() {
        let access = AccessState::default();
        let first = Arc::new(Completion::default());
        let second = Arc::new(Completion::default());
        access.register(&first);
        access.register(&second);
        let worker = std::thread::spawn(move || {
            first.finish(Ok(()));
            second.finish(Ok(()));
        });
        access.synchronize();
        worker.join().unwrap();
        assert!(access.0.borrow().is_empty());
    }
}
