use std::sync::{Arc, atomic::Ordering};

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

    /// Waits for every GPU access registered before this call.
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

    #[cfg(test)]
    pub(super) fn is_idle(&self) -> bool {
        lock(&self.pending).is_empty()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::SubmissionReport;

    #[test]
    fn buffer_access_waits_for_every_pending_submission() {
        let access = AccessState::default();
        let first = Arc::new(Completion::default());
        let second = Arc::new(Completion::default());
        access.register(&first);
        access.register(&second);
        let worker = std::thread::spawn(move || {
            first.finish(Ok(SubmissionReport::default()));
            second.finish(Ok(SubmissionReport::default()));
        });
        access.synchronize();
        worker.join().unwrap();
        assert!(access.is_idle());
    }
}
