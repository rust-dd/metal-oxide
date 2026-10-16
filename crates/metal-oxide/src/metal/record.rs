use std::{
    ops::Range,
    sync::{Arc, Weak},
};

use super::{
    Argument,
    access::{Access, Conflict, Reservation},
    argument::{ArgumentInfo, Binding},
    buffer::Resource,
    completion::Completion,
    failures::Failures,
    pipeline::Source,
    queue::Signal,
};
use crate::{DynamicLaunchConfig, Result};

/// A dispatch that a batch encodes when it commits.
pub(super) struct Dispatch {
    pub(super) source: Source,
    pub(super) config: DynamicLaunchConfig,
    pub(super) bindings: Range<usize>,
    pub(super) arguments: Vec<ArgumentInfo>,
    /// The element counts of the arguments, for kernels that read buffer lengths.
    pub(super) lengths: Range<usize>,
    /// Position among the batch's dispatches in push order.
    pub(super) index: usize,
    /// The barrier-separated level it runs in: the first after every level that holds a
    /// conflicting access to one of its buffers.
    pub(super) level: usize,
}

/// A buffer the batch uses, with the level of its latest write and of its reads since.
struct Tracked {
    resource: Resource,
    reservation: Reservation,
    write: usize,
    read: usize,
}

/// The dispatches pushed into a batch, with what the native encoders bind.
#[derive(Default)]
pub(super) struct Recording {
    pub(super) dispatches: Vec<Dispatch>,
    pub(super) bindings: Vec<Binding>,
    /// Canonical bytes of value arguments.
    pub(super) store: Vec<u8>,
    pub(super) lengths: Vec<u64>,
    tracked: Vec<Tracked>,
}

impl Recording {
    /// Records a dispatch. A read follows the latest write of its buffer; a write
    /// follows that write and every read since.
    pub(super) fn record(
        &mut self,
        source: Source,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
    ) -> Result<()> {
        let bindings = self.bindings.len();
        if config.is_empty() {
            self.dispatches.push(Dispatch {
                source,
                config,
                bindings: bindings..bindings,
                arguments: arguments.iter().map(Argument::info).collect(),
                lengths: self.lengths.len()..self.lengths.len(),
                index: self.dispatches.len(),
                level: 0,
            });
            return Ok(());
        }
        let mut resources = Vec::<Resource>::new();
        for resource in arguments.iter().filter_map(Argument::resource) {
            if let Some(known) = resources
                .iter_mut()
                .find(|known| Arc::ptr_eq(&known.access, &resource.access))
            {
                known.read |= resource.read;
                known.written |= resource.written;
            } else {
                resources.push(resource);
            }
        }
        let mut reservations = Vec::with_capacity(resources.len());
        let mut upgrades = Vec::new();
        let mut level = 1;
        for resource in &resources {
            if let Some(tracked) = self
                .tracked
                .iter()
                .find(|tracked| Arc::ptr_eq(&tracked.resource.access, &resource.access))
            {
                upgrades.push(tracked.reservation.upgrade(resource.written)?);
                reservations.push(None);
                let after = if resource.written {
                    tracked.write.max(tracked.read)
                } else {
                    tracked.write
                };
                level = level.max(after + 1);
            } else {
                reservations.push(Some(resource.access.reserve(resource.written)?));
            }
        }
        for argument in arguments {
            argument.prepare();
            self.bindings.push(argument.binding(&mut self.store));
        }
        let lengths = self.lengths.len();
        self.lengths.extend(
            arguments
                .iter()
                .map(|argument| argument.length().unwrap_or(0) as u64),
        );
        for upgrade in upgrades {
            upgrade.keep();
        }
        for (resource, reservation) in resources.into_iter().zip(reservations) {
            let (read, written) = (resource.read, resource.written);
            let index = match self
                .tracked
                .iter()
                .position(|tracked| Arc::ptr_eq(&tracked.resource.access, &resource.access))
            {
                Some(index) => index,
                None => {
                    self.tracked.push(Tracked {
                        resource,
                        reservation: reservation.expect("newly tracked buffer is reserved"),
                        write: 0,
                        read: 0,
                    });
                    self.tracked.len() - 1
                }
            };
            let tracked = &mut self.tracked[index];
            tracked.resource.read |= read;
            if written {
                tracked.resource.written = true;
                tracked.write = level;
                tracked.read = 0;
            } else {
                tracked.read = tracked.read.max(level);
            }
        }
        self.dispatches.push(Dispatch {
            source,
            config,
            bindings: bindings..self.bindings.len(),
            arguments: arguments.iter().map(Argument::info).collect(),
            lengths: lengths..self.lengths.len(),
            index: self.dispatches.len(),
            level,
        });
        Ok(())
    }

    /// The dispatches in encoding order: by level, then in push order.
    pub(super) fn schedule(&self) -> Vec<&Dispatch> {
        let mut order = self.dispatches.iter().collect::<Vec<_>>();
        order.sort_by_key(|dispatch| dispatch.level);
        order
    }

    /// Whether each level holds a single dispatch, so every dispatch waits for the previous.
    /// Every level up to the highest holds a dispatch.
    pub(super) fn is_chain(&self) -> bool {
        self.dispatches.iter().map(|dispatch| dispatch.level).max() == Some(self.dispatches.len())
    }

    /// The unfinished accesses of other batches that this batch must follow, and the failure
    /// words of those whose writes it reads.
    pub(super) fn conflicts(&self) -> (Vec<Conflict>, Vec<Arc<Failures>>) {
        let mut conflicts = Vec::new();
        let mut upstream = Vec::<Arc<Failures>>::new();
        for tracked in &self.tracked {
            let start = conflicts.len();
            tracked
                .resource
                .access
                .conflicts(tracked.resource.written, &mut conflicts);
            if !tracked.resource.read {
                continue;
            }
            for conflict in &conflicts[start..] {
                if let Some(failures) = conflict
                    .failures
                    .as_ref()
                    .filter(|_| conflict.write)
                    .and_then(Weak::upgrade)
                    && !upstream.iter().any(|known| Arc::ptr_eq(known, &failures))
                {
                    upstream.push(failures);
                }
            }
        }
        (conflicts, upstream)
    }

    /// Records the batch's accesses, so CPU views and later batches wait for its command buffer.
    pub(super) fn register(
        &mut self,
        completion: &Arc<Completion>,
        queue: usize,
        signal: Option<&Signal>,
        failures: &Arc<Failures>,
    ) {
        let failures = failures.has_words().then(|| Arc::downgrade(failures));
        for tracked in &mut self.tracked {
            let write = tracked.resource.written;
            tracked.reservation.register(Access {
                completion: Arc::clone(completion),
                queue,
                signal: signal.cloned(),
                write,
                failures: failures.clone().filter(|_| write),
            });
        }
    }
}
