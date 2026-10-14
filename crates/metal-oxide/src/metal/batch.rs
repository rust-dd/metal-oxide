use std::sync::Arc;

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLSize};

use super::{
    Argument, Device, Operation, Pipeline,
    argument::Binding,
    backend::Backend,
    classic::ClassicBatch,
    completion::Completion,
    failures::{Failures, Gate, Segment, Slot},
    metal4::Metal4Batch,
    queue::{Finish, Placement, Signal},
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
                let (conflicts, upstream) = self.recording.conflicts();
                let placement = Placement::new(&device.backend().running(), conflicts);
                let encoder = Encoder::new(&device, &self.recording, &placement, upstream)?;
                let signal = match device.backend() {
                    Backend::Classic { signals, .. } => Some(signals.take(device.raw())?),
                    Backend::Metal4 { .. } => None,
                };
                let failures = Arc::new(encoder.failures);
                self.recording
                    .register(&completion, placement.queue, signal.as_ref(), &failures);
                encoder.native.commit(
                    &device,
                    &completion,
                    &placement,
                    signal,
                    encoder.pipelines,
                    failures,
                );
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
    /// Encodes the recorded dispatches level by level, with a barrier between levels, on the
    /// queue of `placement`, after the command buffers in `upstream` whose writes they read.
    fn new(
        device: &Device,
        recording: &Recording,
        placement: &Placement,
        upstream: Vec<Arc<Failures>>,
    ) -> Result<Self> {
        let native = match device.backend() {
            Backend::Classic { queues, .. } => {
                let mut waits = Vec::<Signal>::new();
                for access in &placement.remote {
                    let signal = access
                        .signal
                        .as_ref()
                        .expect("classic accesses carry a signal");
                    if !waits.iter().any(|wait| {
                        std::ptr::eq(&*wait.event, &*signal.event) && wait.value == signal.value
                    }) {
                        waits.push(signal.clone());
                    }
                }
                NativeBatch::Classic(ClassicBatch::new(
                    &queues[placement.queue].native,
                    device.log_state(),
                    recording.is_chain(),
                    &waits,
                )?)
            }
            Backend::Metal4 { queues, cache, .. } => NativeBatch::Metal4(Metal4Batch::new(
                device.raw(),
                &queues[placement.queue].native.raw,
                cache,
                device.log_state(),
                placement.local,
            )?),
        };
        let mut encoder = Self {
            native,
            failures: Failures::new(Arc::clone(device.failure_pool())),
            pipelines: Vec::new(),
        };
        if let Some(Gate::Snapshot { gate, sources }) =
            encoder.failures.follow(device.raw(), upstream)?
        {
            encoder.snapshot(device, gate, &sources)?;
        }
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
        match self.failures.gate(device.raw())? {
            Gate::Word(slot) => Ok(slot),
            Gate::Snapshot { gate, sources } => {
                self.snapshot(device, gate, &sources)?;
                Ok(gate)
            }
        }
    }

    /// Sets `gate` when any word of `sources` is set, before every later dispatch.
    fn snapshot(&mut self, device: &Device, gate: Slot, sources: &[Segment]) -> Result<()> {
        let snapshot = device.snapshot()?;
        for segment in sources {
            let words = [
                Word {
                    index: 0,
                    buffer: &segment.buffer,
                    offset: segment.offset,
                },
                Word::new(&self.failures, 1, gate),
            ];
            let config = DynamicLaunchConfig::new(Dim3::x(1), Dim3::x(segment.count as u32));
            // SAFETY: each thread reads one word of finished work and may set the fresh gate;
            // `Failures` retains both chunks through completion.
            unsafe {
                self.native
                    .launch(device, snapshot, config, &[], &[], &words)?
            };
        }
        self.pipelines.push(snapshot.clone());
        self.native.barrier();
        Ok(())
    }
}

impl NativeBatch {
    fn commit(
        self,
        device: &Device,
        completion: &Arc<Completion>,
        placement: &Placement,
        signal: Option<Signal>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        failures: Arc<Failures>,
    ) {
        match (self, device.backend(), signal) {
            (NativeBatch::Classic(batch), Backend::Classic { queues, signals }, Some(signal)) => {
                let finish = Finish::new(
                    queues[placement.queue].start(),
                    Some((signal.clone(), Arc::clone(signals))),
                );
                batch.commit(Arc::clone(completion), pipelines, failures, &signal, finish)
            }
            (NativeBatch::Metal4(batch), Backend::Metal4 { queues, cache, .. }, None) => {
                let mut remote = placement
                    .remote
                    .iter()
                    .map(|access| access.queue)
                    .collect::<Vec<_>>();
                remote.sort_unstable();
                remote.dedup();
                let waits = remote
                    .into_iter()
                    .map(|queue| queues[queue].native.signal())
                    .collect::<Vec<_>>();
                let finish = Finish::new(queues[placement.queue].start(), None);
                batch.commit(
                    cache,
                    Arc::clone(completion),
                    pipelines,
                    failures,
                    &waits,
                    finish,
                )
            }
            _ => unreachable!("a device keeps its native command path"),
        }
    }

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
