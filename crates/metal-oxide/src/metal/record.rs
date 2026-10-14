use std::{
    ops::Range,
    sync::{Arc, Weak},
};

use super::{
    Argument, Pipeline,
    access::{Access, AccessState, Conflict},
    argument::Binding,
    buffer::Resource,
    completion::Completion,
    failures::Failures,
    queue::Signal,
};
use crate::DynamicLaunchConfig;

/// A dispatch that a batch encodes when it commits.
pub(super) struct Dispatch {
    pub(super) pipeline: Pipeline,
    pub(super) config: DynamicLaunchConfig,
    pub(super) bindings: Range<usize>,
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
    /// Records a validated dispatch. A read follows the latest write of its buffer; a write
    /// follows that write and every read since.
    pub(super) fn record(
        &mut self,
        pipeline: &Pipeline,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
    ) {
        let bindings = self.bindings.len();
        let mut resources = Vec::new();
        let mut level = 1;
        for argument in arguments {
            argument.prepare();
            self.bindings.push(argument.binding(&mut self.store));
            if let Some(resource) = argument.resource() {
                if let Some(tracked) = self.find(&resource.access) {
                    let after = if resource.written {
                        tracked.write.max(tracked.read)
                    } else {
                        tracked.write
                    };
                    level = level.max(after + 1);
                }
                resources.push(resource);
            }
        }
        let lengths = self.lengths.len();
        if pipeline.checks().is_some_and(|kernel| kernel.lengths) {
            self.lengths.extend(
                arguments
                    .iter()
                    .map(|argument| argument.length().unwrap_or(0) as u64),
            );
        }
        for resource in resources {
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
            pipeline: pipeline.clone(),
            config,
            bindings: bindings..self.bindings.len(),
            lengths: lengths..self.lengths.len(),
            index: self.dispatches.len(),
            level,
        });
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
        &self,
        completion: &Arc<Completion>,
        queue: usize,
        signal: Option<&Signal>,
        failures: &Arc<Failures>,
    ) {
        let failures = failures.has_words().then(|| Arc::downgrade(failures));
        for tracked in &self.tracked {
            let write = tracked.resource.written;
            tracked.resource.access.register(Access {
                completion: Arc::clone(completion),
                queue,
                signal: signal.cloned(),
                write,
                failures: failures.clone().filter(|_| write),
            });
            if tracked.resource.written {
                tracked.resource.access.mark_written();
            }
        }
    }

    fn find(&self, access: &Arc<AccessState>) -> Option<&Tracked> {
        self.tracked
            .iter()
            .find(|tracked| Arc::ptr_eq(&tracked.resource.access, access))
    }
}
