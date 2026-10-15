use std::{
    sync::atomic::Ordering,
    task::{Context, Poll, Waker},
    time::Duration,
};

use super::{
    failure::CommandFailure,
    sync::{AtomicBool, Condvar, Mutex, MutexGuard, lock},
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
    /// Set once the result is published, so checks need not lock.
    finished: AtomicBool,
}

impl Completion {
    pub(super) fn finish(&self, result: std::result::Result<GpuTime, CommandFailure>) {
        let wakers = {
            let mut state = lock(&self.state);
            if state.result.is_some() {
                return;
            }
            state.result = Some(result);
            self.finished.store(true, Ordering::Release);
            std::mem::take(&mut state.wakers)
        };
        self.ready.notify_all();
        for waker in wakers {
            // A user waker must not unwind through the native completion callback.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }

    pub(super) fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
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
mod tests;
