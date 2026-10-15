use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTL4CommandQueue, MTLDevice, MTLSharedEvent};

use super::{
    access::Conflict,
    sync::{Mutex, lock},
};
use crate::{Error, Result};

/// The command queues a device opens. Completions arrive in commit order per queue, so
/// operations without shared buffers on different queues finish independently.
pub(super) const QUEUES: usize = 4;

/// A native command queue and the number of batches in flight on it.
pub(super) struct Queue<Q> {
    pub(super) native: Q,
    running: Arc<AtomicUsize>,
}

impl<Q> Queue<Q> {
    pub(super) fn new(native: Q) -> Self {
        Self {
            native,
            running: Arc::default(),
        }
    }

    /// Counts a batch committed to this queue until its completion handler runs `Finish`.
    pub(super) fn start(&self) -> Arc<AtomicUsize> {
        self.running.fetch_add(1, Ordering::Relaxed);
        Arc::clone(&self.running)
    }

    pub(super) fn running(&self) -> usize {
        self.running.load(Ordering::Relaxed)
    }
}

/// The queue a batch runs on and the conflicting work in flight that it must follow.
pub(super) struct Placement {
    pub(super) queue: usize,
    /// Conflicting work runs on the batch's own queue.
    pub(super) local: bool,
    /// Conflicting accesses on other queues.
    pub(super) remote: Vec<Conflict>,
}

impl Placement {
    /// The queue with the most conflicting accesses, where the queue itself orders them, or
    /// without conflicts the queue with the fewest batches in flight.
    pub(super) fn new(running: &[usize; QUEUES], conflicts: &[Conflict]) -> Self {
        let mut counts = [0_usize; QUEUES];
        for conflict in conflicts {
            counts[conflict.queue] += 1;
        }
        let queue = match (0..QUEUES).max_by_key(|queue| counts[*queue]) {
            Some(queue) if counts[queue] > 0 => queue,
            _ => (0..QUEUES).min_by_key(|queue| running[*queue]).unwrap_or(0),
        };
        Self {
            queue,
            local: counts[queue] > 0,
            remote: conflicts
                .iter()
                .filter(|conflict| conflict.queue != queue)
                .cloned()
                .collect(),
        }
    }
}

/// An event value that a classic command buffer sets when its work finishes.
#[derive(Clone)]
pub(super) struct Signal {
    pub(super) event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    pub(super) value: u64,
}

/// Events of finished classic command buffers, reused with their next value.
#[derive(Default)]
pub(super) struct Signals(Mutex<Vec<Signal>>);

impl Signals {
    pub(super) fn take(&self, device: &ProtocolObject<dyn MTLDevice>) -> Result<Signal> {
        if let Some(Signal { event, value }) = lock(&self.0).pop() {
            return Ok(Signal {
                event,
                value: value + 1,
            });
        }
        let event = device
            .newSharedEvent()
            .ok_or_else(|| Error::Command("could not create Metal shared event".into()))?;
        Ok(Signal { event, value: 1 })
    }

    /// Sets an event that a failed command buffer left unset, so no waiter blocks on it, and
    /// keeps it for reuse.
    fn put(&self, signal: Signal) {
        if signal.event.signaledValue() < signal.value {
            signal.event.setSignaledValue(signal.value);
        }
        lock(&self.0).push(signal);
    }
}

/// What a command buffer's completion handler releases: its queue's running count and, for
/// classic, the event it set.
pub(super) struct Finish {
    running: Arc<AtomicUsize>,
    signal: Option<(Signal, Arc<Signals>)>,
}

impl Finish {
    pub(super) fn new(running: Arc<AtomicUsize>, signal: Option<(Signal, Arc<Signals>)>) -> Self {
        Self { running, signal }
    }

    /// Runs once, from the completion handler.
    pub(super) fn run(&self) {
        if let Some((signal, signals)) = &self.signal {
            signals.put(signal.clone());
        }
        self.running.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A Metal 4 queue with a timeline event that it signals on demand.
pub(super) struct Metal4Queue {
    pub(super) raw: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    timeline: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    signaled: Mutex<u64>,
}

impl Metal4Queue {
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let raw = device
            .newMTL4CommandQueue()
            .ok_or_else(|| Error::Command("could not create Metal 4 command queue".into()))?;
        let timeline = device
            .newSharedEvent()
            .ok_or_else(|| Error::Command("could not create Metal shared event".into()))?;
        Ok(Self {
            raw,
            timeline,
            signaled: Mutex::new(0),
        })
    }

    /// Signals the timeline once every batch committed to this queue so far has finished;
    /// the value is set in call order, so it only grows.
    pub(super) fn signal(&self) -> (&ProtocolObject<dyn MTLSharedEvent>, u64) {
        let mut value = lock(&self.signaled);
        *value += 1;
        self.raw
            .signalEvent_value(ProtocolObject::from_ref(&*self.timeline), *value);
        (&self.timeline, *value)
    }
}

#[cfg(test)]
mod tests;
