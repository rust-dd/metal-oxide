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
mod tests {
    use super::*;

    fn conflict(queue: usize, write: bool) -> Conflict {
        Conflict {
            queue,
            write,
            signal: None,
            failures: None,
        }
    }

    #[test]
    fn dependent_batches_join_the_queue_of_their_conflicts() {
        let placement = Placement::new(&[3, 0, 0, 0], &[conflict(2, true), conflict(2, false)]);
        assert_eq!((placement.queue, placement.local), (2, true));
        assert!(placement.remote.is_empty());
        let placement = Placement::new(
            &[0, 0, 0, 0],
            &[conflict(1, true), conflict(3, false), conflict(3, true)],
        );
        assert_eq!((placement.queue, placement.local), (3, true));
        assert_eq!(
            placement
                .remote
                .iter()
                .map(|access| access.queue)
                .collect::<Vec<_>>(),
            [1]
        );
    }

    #[test]
    fn independent_batches_go_to_the_least_busy_queue() {
        let placement = Placement::new(&[2, 1, 0, 1], &[]);
        assert_eq!((placement.queue, placement.local), (2, false));
        assert!(placement.remote.is_empty());
        assert_eq!(Placement::new(&[0, 0, 0, 0], &[]).queue, 0);
    }

    #[test]
    #[ignore = "requires a Metal device on Apple Silicon"]
    fn independent_operations_run_on_separate_queues() -> crate::Result<()> {
        use crate::{Argument, Launch, LaunchConfig, Module, Pipeline};
        use std::{future::IntoFuture, pin::Pin, task::Context};

        let device = crate::Device::system_default()?;
        let module = Module::from_source(
            &device,
            "#include <metal_stdlib>
            using namespace metal;
            kernel void spin(device float* data [[buffer(0)]], uint i [[thread_position_in_grid]]) {
                float x = data[i];
                for (uint k = 0; k < 2000000; ++k) { x = fma(x, 0.999f, 0.001f); }
                data[i] = x;
            }",
        )?;
        let pipeline = Pipeline::new(&device, &module, "spin")?;
        let mut buffers = [
            device.buffer_zeroed::<f32>(256)?,
            device.buffer_zeroed(256)?,
        ];
        let config = LaunchConfig::<256>::for_elements(256)?;
        let mut running = Vec::new();
        for buffer in &mut buffers {
            // SAFETY: each thread updates its own element of a 256-element buffer.
            let mut future = unsafe {
                Launch::new(Ok(&pipeline), config, (buffer,), |(buffer,)| {
                    Ok([Argument::read_write(buffer)])
                })
            }
            .into_future();
            let _ = Pin::new(&mut future).poll(&mut Context::from_waker(std::task::Waker::noop()));
            running.push(device.backend().running());
            drop(future);
        }
        assert_eq!(running[0].iter().sum::<usize>(), 1);
        assert_eq!(
            running[1].iter().filter(|count| **count == 1).count(),
            2,
            "{running:?}"
        );
        Ok(())
    }
}
