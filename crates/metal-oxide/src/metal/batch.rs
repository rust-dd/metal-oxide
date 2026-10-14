use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLSize};

use super::{
    Argument, Device, Operation, Pipeline,
    argument::Binding,
    backend::Backend,
    classic::ClassicBatch,
    completion::Completion,
    failures::{Failures, Gate, Slot},
    metal4::Metal4Batch,
    record::Recording,
};
use crate::{Dim3, DynamicLaunchConfig, Error, Result};

/// The command buffer an operation encodes into; see [`sequence`](super::sequence).
///
/// A batch records its dispatches and encodes them when it commits. A dispatch runs after the
/// earlier ones whose buffers it reads or writes; dispatches without such conflicts may run
/// concurrently.
pub struct Batch {
    device: Option<Device>,
    recording: Recording,
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
            recording: Recording::default(),
        }
    }

    /// Encodes `operation` and returns its output, which is ready to use as the arguments of
    /// later operations in the batch.
    pub fn push<O: Operation>(&mut self, operation: O) -> Result<O::Output> {
        operation.encode(self)
    }

    /// Records a kernel dispatch after validating it.
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
        if !config.is_empty() {
            self.recording.record(pipeline, config, arguments);
        }
        Ok(())
    }

    /// Encodes and commits the recorded dispatches; a batch without any finishes immediately.
    pub(super) fn commit(self) -> Result<Arc<Completion>> {
        let completion = Arc::new(Completion::default());
        match self.device {
            Some(device) if !self.recording.dispatches.is_empty() => {
                let encoder = Encoder::new(&device, &self.recording)?;
                self.recording.register(&completion);
                encoder.commit(&device, &completion);
            }
            _ => completion.finish(Ok(None)),
        }
        Ok(completion)
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

/// The native command buffer of a committing batch.
struct Encoder {
    native: NativeBatch,
    failures: Failures,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

enum NativeBatch {
    Classic(ClassicBatch),
    Metal4(Metal4Batch),
}

impl Encoder {
    /// Encodes the recorded dispatches level by level, with a barrier between levels.
    fn new(device: &Device, recording: &Recording) -> Result<Self> {
        let native = match device.backend() {
            Backend::Classic(queue) => NativeBatch::Classic(ClassicBatch::new(
                queue,
                device.log_state(),
                recording.is_chain(),
            )?),
            Backend::Metal4 { queue, cache, .. } => NativeBatch::Metal4(Metal4Batch::new(
                device.raw(),
                queue,
                cache,
                device.log_state(),
            )?),
        };
        let mut encoder = Self {
            native,
            failures: Failures::new(Arc::clone(device.failure_pool())),
            pipelines: Vec::new(),
        };
        let schedule = recording.schedule();
        for (position, level) in schedule
            .chunk_by(|first, second| first.level == second.level)
            .enumerate()
        {
            if position > 0 {
                encoder.native.barrier();
                encoder.failures.close_level();
            }
            let gated = level
                .iter()
                .any(|dispatch| dispatch.pipeline.checks().is_some());
            let gate = if gated {
                Some(encoder.gate(device)?)
            } else {
                None
            };
            for dispatch in level {
                let mut slots = Vec::new();
                if let (Some(kernel), Some(gate)) = (dispatch.pipeline.checks(), gate) {
                    let binding = kernel.gate_binding as usize;
                    slots.push((binding, gate));
                    if kernel.fallible {
                        let word =
                            encoder
                                .failures
                                .failure_word(device.raw(), kernel, dispatch.index)?;
                        slots.push((binding + 1, word));
                    }
                    if kernel.lengths {
                        let lengths = &recording.lengths[dispatch.lengths.clone()];
                        let table = encoder.failures.table(device.raw(), lengths)?;
                        slots.push((binding + 1 + usize::from(kernel.fallible), table));
                    }
                }
                let words = slots
                    .into_iter()
                    .map(|(index, slot)| Word::new(&encoder.failures, index, slot))
                    .collect::<Vec<_>>();
                let bindings = &recording.bindings[dispatch.bindings.clone()];
                // SAFETY: validation passed when the dispatch was recorded; its pusher provides
                // the shader safety contract.
                unsafe {
                    encoder.native.launch(
                        device,
                        dispatch.pipeline.raw(),
                        dispatch.config,
                        bindings,
                        &recording.store,
                        &words,
                    )?
                };
                encoder.pipelines.push(dispatch.pipeline.raw().clone());
            }
        }
        Ok(encoder)
    }

    /// The gate of a level's generated dispatches, set by snapshots if it covers several words.
    fn gate(&mut self, device: &Device) -> Result<Slot> {
        let (gate, sources) = match self.failures.gate(device.raw())? {
            Gate::Word(slot) => return Ok(slot),
            Gate::Snapshot { gate, sources } => (gate, sources),
        };
        let snapshot = device.snapshot()?;
        for segment in &sources {
            let words = [
                Word::new(&self.failures, 0, segment.start),
                Word::new(&self.failures, 1, gate),
            ];
            let config = DynamicLaunchConfig::new(Dim3::x(1), Dim3::x(segment.count as u32));
            // SAFETY: each thread reads one word that ended levels wrote, and may set the
            // fresh gate; both stay retained by `Failures` through completion.
            unsafe {
                self.native
                    .launch(device, snapshot, config, &[], &[], &words)?
            };
        }
        self.pipelines.push(snapshot.clone());
        self.native.barrier();
        Ok(gate)
    }

    fn commit(self, device: &Device, completion: &Arc<Completion>) {
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
        bindings: &[Binding],
        store: &[u8],
        words: &[Word<'_>],
    ) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe {
            match self {
                Self::Classic(batch) => {
                    batch.launch(device.raw(), pipeline, config, bindings, store, words)
                }
                Self::Metal4(batch) => {
                    batch.launch(device.raw(), pipeline, config, bindings, store, words)
                }
            }
        }
    }

    fn barrier(&self) {
        match self {
            Self::Classic(batch) => batch.barrier(),
            Self::Metal4(batch) => batch.barrier(),
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
