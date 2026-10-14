use std::{ops::Range, sync::Arc};

use super::{
    Argument, Pipeline,
    access::{Access, AccessState},
    argument::Binding,
    buffer::Resource,
    completion::Completion,
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
            let written = resource.written;
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

    /// The unfinished accesses of other batches that this batch must follow.
    pub(super) fn conflicts(&self) -> Vec<Access> {
        let mut conflicts = Vec::new();
        for tracked in &self.tracked {
            tracked
                .resource
                .access
                .conflicts(tracked.resource.written, &mut conflicts);
        }
        conflicts
    }

    /// Records the batch's accesses, so CPU views and later batches wait for its command buffer.
    pub(super) fn register(
        &self,
        completion: &Arc<Completion>,
        queue: usize,
        signal: Option<&Signal>,
    ) {
        for tracked in &self.tracked {
            tracked.resource.access.register(Access {
                completion: Arc::clone(completion),
                queue,
                signal: signal.cloned(),
                write: tracked.resource.written,
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
