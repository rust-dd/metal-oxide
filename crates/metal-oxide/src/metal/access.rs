use std::{
    sync::{Arc, Weak, atomic::Ordering},
    task::{Context, Poll},
};

use super::{
    completion::Completion,
    failures::Failures,
    queue::Signal,
    sync::{AtomicBool, Mutex, lock},
};

/// A committed GPU access to a buffer.
#[derive(Clone)]
pub(super) struct Access {
    pub(super) completion: Arc<Completion>,
    pub(super) queue: usize,
    /// The event a classic command buffer sets when its work finishes.
    pub(super) signal: Option<Signal>,
    pub(super) write: bool,
    /// The failure words of a write, which later readers gate on while it runs.
    pub(super) failures: Option<Weak<Failures>>,
}

/// GPU work registered on one buffer, tracked independently of the futures that observe it.
#[derive(Default)]
pub(super) struct AccessState {
    pending: Mutex<Vec<Access>>,
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

    pub(super) fn register(&self, access: Access) {
        let mut pending = lock(&self.pending);
        pending.retain(|access| !access.completion.is_finished());
        pending.push(access);
    }

    /// The unfinished accesses that a new access must follow: every one before a write, the
    /// writes before a read.
    pub(super) fn conflicts(&self, write: bool, conflicts: &mut Vec<Access>) {
        let mut pending = lock(&self.pending);
        pending.retain(|access| !access.completion.is_finished());
        conflicts.extend(
            pending
                .iter()
                .filter(|access| write || access.write)
                .cloned(),
        );
    }

    /// Blocks until every GPU access registered before this call has finished.
    pub(super) fn synchronize(&self) {
        let snapshot = {
            let mut pending = lock(&self.pending);
            pending.retain(|access| !access.completion.is_finished());
            if pending.is_empty() {
                return;
            }
            pending
                .iter()
                .map(|access| Arc::clone(&access.completion))
                .collect::<Vec<_>>()
        };
        // Waiting without the lock lets other threads register and synchronize meanwhile.
        for completion in &snapshot {
            completion.block();
        }
        lock(&self.pending).retain(|access| !access.completion.is_finished());
    }

    /// Whether every registered GPU access has finished; otherwise wakes `cx` when the
    /// earliest unfinished one does.
    pub(super) fn poll(&self, cx: &mut Context<'_>) -> Poll<()> {
        let mut pending = lock(&self.pending);
        loop {
            pending.retain(|access| !access.completion.is_finished());
            match pending.first() {
                None => return Poll::Ready(()),
                Some(access) if access.completion.poll_finished(cx).is_pending() => {
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

/// A classic write on the first queue, for tests of CPU synchronization.
#[cfg(test)]
pub(super) fn write(completion: &Arc<Completion>) -> Access {
    Access {
        completion: Arc::clone(completion),
        queue: 0,
        signal: None,
        write: true,
        failures: None,
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use std::task::Waker;

    use super::*;

    #[test]
    fn buffer_access_waits_for_every_pending_command_buffer() {
        let access = AccessState::default();
        let first = Arc::new(Completion::default());
        let second = Arc::new(Completion::default());
        access.register(write(&first));
        access.register(write(&second));
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
        access.register(write(&first));
        access.register(write(&second));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(access.poll(&mut cx).is_pending());
        first.finish(Ok(None));
        assert!(access.poll(&mut cx).is_pending());
        second.finish(Ok(None));
        assert!(access.poll(&mut cx).is_ready());
        assert!(access.is_idle());
    }
}
