use std::{
    sync::{Arc, atomic::Ordering},
    task::{Context, Poll},
};

use super::{
    completion::Completion,
    sync::{AtomicBool, Mutex, lock},
};

/// GPU work registered on one buffer, tracked independently of the futures that observe it.
#[derive(Default)]
pub(super) struct AccessState {
    pending: Mutex<Vec<Arc<Completion>>>,
    written: AtomicBool,
}

impl AccessState {
    /// Records a committed GPU write so the next CPU read decodes the GPU copy.
    pub(super) fn mark_written(&self) {
        self.written.store(true, Ordering::Release);
    }

    pub(super) fn take_written(&self) -> bool {
        self.written.swap(false, Ordering::AcqRel)
    }

    pub(super) fn register(&self, completion: &Arc<Completion>) {
        let mut pending = lock(&self.pending);
        pending.retain(|completion| !completion.is_finished());
        pending.push(Arc::clone(completion));
    }

    /// Blocks until every GPU access registered before this call has finished.
    pub(super) fn synchronize(&self) {
        let snapshot = {
            let mut pending = lock(&self.pending);
            pending.retain(|completion| !completion.is_finished());
            if pending.is_empty() {
                return;
            }
            pending.clone()
        };
        // Waiting without the lock lets other threads register and synchronize meanwhile.
        for completion in &snapshot {
            completion.block();
        }
        lock(&self.pending).retain(|completion| !completion.is_finished());
    }

    /// Whether every registered GPU access has finished; otherwise wakes `cx` when the
    /// earliest unfinished one does.
    pub(super) fn poll(&self, cx: &mut Context<'_>) -> Poll<()> {
        let mut pending = lock(&self.pending);
        loop {
            pending.retain(|completion| !completion.is_finished());
            match pending.first() {
                None => return Poll::Ready(()),
                Some(completion) if completion.poll_finished(cx).is_pending() => {
                    return Poll::Pending;
                }
                Some(_) => {}
            }
        }
    }

    #[cfg(test)]
    pub(super) fn is_idle(&self) -> bool {
        lock(&self.pending).is_empty()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::task::Waker;

    use super::*;

    #[test]
    fn buffer_access_waits_for_every_pending_submission() {
        let access = AccessState::default();
        let first = Arc::new(Completion::default());
        let second = Arc::new(Completion::default());
        access.register(&first);
        access.register(&second);
        let worker = std::thread::spawn(move || {
            first.finish(Ok(None));
            second.finish(Ok(None));
        });
        access.synchronize();
        worker.join().unwrap();
        assert!(access.is_idle());
    }

    #[test]
    fn polled_access_is_ready_only_after_every_registered_completion() {
        let access = AccessState::default();
        let first = Arc::new(Completion::default());
        let second = Arc::new(Completion::default());
        access.register(&first);
        access.register(&second);
        let mut cx = Context::from_waker(Waker::noop());
        assert!(access.poll(&mut cx).is_pending());
        first.finish(Ok(None));
        assert!(access.poll(&mut cx).is_pending());
        second.finish(Ok(None));
        assert!(access.poll(&mut cx).is_ready());
        assert!(access.is_idle());
    }
}
