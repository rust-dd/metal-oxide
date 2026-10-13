use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLSize};

use super::{
    Argument, Device, Operation, Pipeline,
    backend::Backend,
    buffer::Resource,
    classic::ClassicBatch,
    completion::Completion,
    failures::{Failures, Slot},
    metal4::Metal4Batch,
};
use crate::{Dim3, DynamicLaunchConfig, Error, Result};

enum NativeBatch {
    Classic(ClassicBatch),
    Metal4(Metal4Batch),
}

/// The command buffer an operation encodes into; see [`sequence`](super::sequence).
///
/// It holds native command encoders, so it stays on the encoding thread:
///
/// ```compile_fail
/// metal_oxide::sequence(|batch| {
///     std::thread::scope(|scope| {
///         scope.spawn(|| drop(batch));
///     });
///     Ok(())
/// });
/// ```
pub struct Batch {
    device: Option<Device>,
    commands: Option<Commands>,
}

struct Commands {
    native: NativeBatch,
    resources: Vec<Resource>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    failures: Failures,
    dispatches: usize,
    /// A native dispatch, including an internal snapshot, precedes the next one.
    encoded: bool,
}

/// A gate or failure word that a dispatch binds after its parameters.
pub(super) struct Word<'a> {
    pub(super) index: usize,
    pub(super) buffer: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) offset: usize,
}

impl<'a> Word<'a> {
    fn new(failures: &'a Failures, index: usize, slot: Slot) -> Self {
        Self {
            index,
            buffer: failures.buffer(slot),
            offset: slot.offset,
        }
    }
}

impl Batch {
    pub(super) fn new() -> Self {
        Self {
            device: None,
            commands: None,
        }
    }

    /// Encodes `operation` after the batch's previous dispatches and returns its output, which
    /// is ready to use as the arguments of later operations in the batch.
    pub fn push<O: Operation>(&mut self, operation: O) -> Result<O::Output> {
        operation.encode(self)
    }

    /// Encodes a kernel after the batch's preceding dispatches.
    ///
    /// # Safety
    ///
    /// The dispatch must satisfy the launch contract of [`Launch::new`](super::Launch::new).
    pub(super) unsafe fn launch(
        &mut self,
        pipeline: &Pipeline,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
    ) -> Result<()> {
        let device = self.device.get_or_insert_with(|| pipeline.device().clone());
        if pipeline.device().id() != device.id()
            || arguments
                .iter()
                .any(|a| a.device_id().is_some_and(|id| id != device.id()))
        {
            return Err(Error::DeviceMismatch);
        }
        validate(device, pipeline, config, arguments)?;
        if config.is_empty() {
            return Ok(());
        }
        let commands = match &mut self.commands {
            Some(commands) => commands,
            None => self.commands.insert(Commands::new(device)?),
        };
        // SAFETY: validation passed; the caller provides the kernel contract.
        unsafe { commands.launch(device, pipeline, config, arguments) }
    }

    /// Commits the encoded dispatches; a batch without any finishes immediately.
    pub(super) fn commit(self) -> Arc<Completion> {
        let completion = Arc::new(Completion::default());
        match (self.device, self.commands) {
            (Some(device), Some(commands)) => commands.commit(&device, &completion),
            _ => completion.finish(Ok(None)),
        }
        completion
    }
}

fn validate(
    device: &Device,
    pipeline: &Pipeline,
    config: DynamicLaunchConfig,
    arguments: &[Argument<'_>],
) -> Result<()> {
    let mut shared = arguments.iter().filter_map(Argument::threadgroup_bytes);
    let shared_count = shared.clone().count();
    let buffer_count = arguments.len() - shared_count;
    if buffer_count > 31 {
        return Err(Error::TooManyArguments(buffer_count));
    }
    if shared_count > 31 {
        return Err(Error::InvalidLaunch("more than 31 threadgroup arguments"));
    }
    let shared_bytes = shared.try_fold(
        pipeline.raw().staticThreadgroupMemoryLength(),
        |total, bytes| total.checked_add(bytes),
    );
    if shared_bytes.is_none_or(|bytes| bytes > device.max_threadgroup_memory_per_block()) {
        return Err(Error::InvalidLaunch(
            "kernel exceeds threadgroup memory capacity",
        ));
    }
    pipeline.validate_launch(config, arguments)?;
    config.validate(
        pipeline.max_threads_per_block(),
        device.max_block_dimensions(),
    )
}

impl Commands {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            native: match device.backend() {
                Backend::Classic(queue) => {
                    NativeBatch::Classic(ClassicBatch::new(queue, device.log_state())?)
                }
                Backend::Metal4 { queue, cache, .. } => NativeBatch::Metal4(Metal4Batch::new(
                    device.raw(),
                    queue,
                    cache,
                    device.log_state(),
                )?),
            },
            resources: Vec::new(),
            pipelines: Vec::new(),
            failures: Failures::new(Arc::clone(device.failure_pool())),
            dispatches: 0,
            encoded: false,
        })
    }

    /// # Safety
    ///
    /// The dispatch must satisfy the launch contract after `validate`.
    unsafe fn launch(
        &mut self,
        device: &Device,
        pipeline: &Pipeline,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
    ) -> Result<()> {
        for argument in arguments {
            argument.prepare();
            if let Some(resource) = argument.resource() {
                if let Some(existing) = self
                    .resources
                    .iter_mut()
                    .find(|existing| Arc::ptr_eq(&existing.access, &resource.access))
                {
                    existing.written |= resource.written;
                } else {
                    self.resources.push(resource);
                }
            }
        }
        let assignment = match pipeline.checks() {
            Some(kernel) => Some((
                kernel.gate_binding as usize,
                self.failures.assign(device.raw(), kernel.fallible)?,
            )),
            None => None,
        };
        if let Some((_, assignment)) = assignment.as_ref().filter(|(_, a)| a.snapshot) {
            let snapshot = device.snapshot()?;
            let words = [
                Word::new(&self.failures, 0, self.failures.failure_slot()),
                Word::new(&self.failures, 1, assignment.gate),
            ];
            self.pipelines.push(snapshot.clone());
            let previous = std::mem::replace(&mut self.encoded, true);
            // SAFETY: one thread copies a word between live, retained failure-storage slots.
            unsafe {
                self.native.launch(
                    device,
                    snapshot,
                    DynamicLaunchConfig::new(Dim3::x(1), Dim3::x(1)),
                    &[],
                    &words,
                    previous,
                )?
            };
        }
        let lengths = match pipeline.checks() {
            Some(kernel) if kernel.lengths => {
                let counts = arguments
                    .iter()
                    .map(|argument| argument.length().unwrap_or(0) as u64)
                    .collect::<Vec<_>>();
                let binding = kernel.gate_binding as usize + 1 + usize::from(kernel.fallible);
                Some((binding, self.failures.table(device.raw(), &counts)?))
            }
            _ => None,
        };
        let words = assignment
            .iter()
            .flat_map(|(binding, assignment)| {
                std::iter::once((*binding, assignment.gate))
                    .chain(assignment.failure.map(|failure| (binding + 1, failure)))
            })
            .chain(lengths)
            .map(|(index, slot)| Word::new(&self.failures, index, slot))
            .collect::<Vec<_>>();
        self.pipelines.push(pipeline.raw().clone());
        let previous = std::mem::replace(&mut self.encoded, true);
        // SAFETY: validation passed; the caller provides the shader safety contract.
        unsafe {
            self.native
                .launch(device, pipeline.raw(), config, arguments, &words, previous)?
        };
        if let (Some(kernel), Some((_, assignment))) = (pipeline.checks(), &assignment)
            && kernel.fallible
        {
            self.failures
                .record(kernel, self.dispatches, assignment.gate);
        }
        self.dispatches += 1;
        Ok(())
    }

    fn commit(self, device: &Device, completion: &Arc<Completion>) {
        for resource in &self.resources {
            resource.access.register(completion);
            if resource.written {
                resource.access.mark_written();
            }
        }
        match (self.native, device.backend()) {
            (NativeBatch::Classic(batch), _) => {
                batch.commit(Arc::clone(completion), self.pipelines, self.failures)
            }
            (NativeBatch::Metal4(batch), Backend::Metal4 { cache, .. }) => {
                batch.commit(cache, Arc::clone(completion), self.pipelines, self.failures)
            }
            (NativeBatch::Metal4(_), Backend::Classic(_)) => {
                unreachable!("a device keeps its native command path")
            }
        }
    }
}

impl NativeBatch {
    /// # Safety
    ///
    /// The dispatch must satisfy the launch contract after `validate`.
    unsafe fn launch(
        &mut self,
        device: &Device,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
        words: &[Word<'_>],
        has_previous_dispatch: bool,
    ) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe {
            match self {
                Self::Classic(batch) => batch.launch(
                    device.raw(),
                    pipeline,
                    config,
                    arguments,
                    words,
                    has_previous_dispatch,
                ),
                Self::Metal4(batch) => batch.launch(
                    device.raw(),
                    pipeline,
                    config,
                    arguments,
                    words,
                    has_previous_dispatch,
                ),
            }
        }
    }
}

pub(super) fn metal_size(dimensions: Dim3) -> MTLSize {
    MTLSize {
        width: dimensions.x as usize,
        height: dimensions.y as usize,
        depth: dimensions.z as usize,
    }
}
