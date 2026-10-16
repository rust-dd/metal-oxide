use std::{
    sync::{Arc, Weak, atomic::Ordering},
    task::{Context, Poll},
};

use super::{
    completion::Completion,
    failures::Failures,
    queue::Signal,
    sync::{AtomicBool, Mutex, MutexGuard, lock},
};
use crate::{Error, Result};

/// A committed GPU access to a buffer.
pub(super) struct Access {
    pub(super) completion: Arc<Completion>,
    pub(super) queue: usize,
    /// The event a classic command buffer sets when its work finishes.
    pub(super) signal: Option<Signal>,
    pub(super) write: bool,
    /// The failure words of a write, which later readers gate on while it runs.
    pub(super) failures: Option<Weak<Failures>>,
}

/// What a new access needs from an earlier one it conflicts with.
#[derive(Clone)]
pub(super) struct Conflict {
    pub(super) queue: usize,
    pub(super) write: bool,
    pub(super) signal: Option<Signal>,
    pub(super) failures: Option<Weak<Failures>>,
}

/// GPU work registered on one buffer, tracked independently of the futures that observe it.
#[derive(Default)]
pub(super) struct AccessState {
    state: Mutex<State>,
    written: AtomicBool,
}

#[derive(Default)]
struct State {
    pending: Vec<Access>,
    recordings: Vec<Recorded>,
}

struct Recorded {
    identity: Arc<()>,
    write: bool,
}

/// Keeps an uncommitted access visible after a launch returns its buffer arguments.
pub(super) struct Reservation {
    access: Arc<AccessState>,
    identity: Arc<()>,
}

/// Restores an existing read reservation when recording a later write fails or unwinds.
pub(super) struct Upgrade<'a> {
    reservation: &'a Reservation,
    restore: bool,
}

impl AccessState {
    pub(super) fn reserve(self: &Arc<Self>, write: bool) -> Result<Reservation> {
        let mut state = lock(&self.state);
        if state
            .recordings
            .iter()
            .any(|recorded| write || recorded.write)
        {
            return Err(Error::InvalidBuffer(
                "buffer has a conflicting uncommitted recording",
            ));
        }
        let identity = Arc::new(());
        state.recordings.push(Recorded {
            identity: Arc::clone(&identity),
            write,
        });
        Ok(Reservation {
            access: Arc::clone(self),
            identity,
        })
    }

    pub(super) fn assert_unrecorded(&self) {
        drop(self.cpu_state(true));
    }

    fn cpu_state(&self, write: bool) -> MutexGuard<'_, State> {
        let state = lock(&self.state);
        if state
            .recordings
            .iter()
            .any(|recorded| write || recorded.write)
        {
            drop(state);
            panic!("CPU access conflicts with an uncommitted buffer recording");
        }
        state
    }

    /// Records a committed GPU write so the next CPU read decodes the GPU copy.
    pub(super) fn mark_written(&self) {
        self.written.store(true, Ordering::Release);
    }

    pub(super) fn take_written(&self) -> bool {
        self.written.swap(false, Ordering::AcqRel)
    }

    #[cfg(test)]
    pub(super) fn register(&self, access: Access) {
        let mut state = lock(&self.state);
        state
            .pending
            .retain(|access| !access.completion.is_finished());
        state.pending.push(access);
    }

    /// The unfinished accesses that a new access must follow: every one before a write, the
    /// writes before a read.
    pub(super) fn conflicts(&self, write: bool, conflicts: &mut Vec<Conflict>) {
        let mut state = lock(&self.state);
        state
            .pending
            .retain(|access| !access.completion.is_finished());
        conflicts.extend(
            state
                .pending
                .iter()
                .filter(|access| write || access.write)
                .map(|access| Conflict {
                    queue: access.queue,
                    write: access.write,
                    signal: access.signal.clone(),
                    failures: access.failures.clone(),
                }),
        );
    }

    /// Blocks until every GPU access registered before this call has finished.
    pub(super) fn synchronize(&self, write: bool) {
        let snapshot = {
            let mut state = self.cpu_state(write);
            state
                .pending
                .retain(|access| !access.completion.is_finished());
            if state.pending.is_empty() {
                return;
            }
            state
                .pending
                .iter()
                .map(|access| Arc::clone(&access.completion))
                .collect::<Vec<_>>()
        };
        // Waiting without the lock lets other threads register and synchronize meanwhile.
        for completion in &snapshot {
            completion.block();
        }
        lock(&self.state)
            .pending
            .retain(|access| !access.completion.is_finished());
    }

    /// Whether every registered GPU access has finished; otherwise wakes `cx` when the
    /// earliest unfinished one does.
    pub(super) fn poll(&self, write: bool, cx: &mut Context<'_>) -> Poll<()> {
        let mut state = self.cpu_state(write);
        loop {
            state
                .pending
                .retain(|access| !access.completion.is_finished());
            match state.pending.first() {
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
        let state = lock(&self.state);
        state.pending.is_empty() && state.recordings.is_empty()
    }
}

impl Reservation {
    pub(super) fn upgrade(&self, write: bool) -> Result<Upgrade<'_>> {
        let mut state = lock(&self.access.state);
        if write
            && state
                .recordings
                .iter()
                .any(|recorded| !Arc::ptr_eq(&recorded.identity, &self.identity))
        {
            return Err(Error::InvalidBuffer(
                "buffer has a conflicting uncommitted recording",
            ));
        }
        let recorded = state
            .recordings
            .iter_mut()
            .find(|recorded| Arc::ptr_eq(&recorded.identity, &self.identity))
            .expect("active recording reservation");
        let restore = write && !recorded.write;
        recorded.write |= write;
        Ok(Upgrade {
            reservation: self,
            restore,
        })
    }

    /// Registers pending work while retaining the reservation until native submission returns.
    pub(super) fn register(&mut self, access: Access) {
        let mut state = lock(&self.access.state);
        state
            .pending
            .retain(|access| !access.completion.is_finished());
        if access.write {
            self.access.mark_written();
        }
        state.pending.push(access);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        lock(&self.access.state)
            .recordings
            .retain(|recorded| !Arc::ptr_eq(&recorded.identity, &self.identity));
    }
}

impl Upgrade<'_> {
    pub(super) fn keep(mut self) {
        self.restore = false;
    }
}

impl Drop for Upgrade<'_> {
    fn drop(&mut self) {
        if self.restore {
            let mut state = lock(&self.reservation.access.state);
            let recorded = state
                .recordings
                .iter_mut()
                .find(|recorded| Arc::ptr_eq(&recorded.identity, &self.reservation.identity))
                .expect("active recording reservation");
            recorded.write = false;
        }
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
mod tests;
