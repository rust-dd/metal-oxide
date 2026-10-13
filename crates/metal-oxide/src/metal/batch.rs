use std::{marker::PhantomData, sync::Arc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLDevice, MTLSize};

use super::{
    Argument, Device, Pipeline, Submission,
    backend::Backend,
    buffer::Resource,
    classic::ClassicBatch,
    completion::{Completion, SubmissionReport},
    failures::{Failures, Slot},
    metal4::Metal4Batch,
};
use crate::{Dim3, DynamicLaunchConfig, Error, Result};

enum NativeBatch<'a> {
    Classic(ClassicBatch),
    Metal4(Metal4Batch<'a>),
}

impl NativeBatch<'_> {
    /// # Safety
    ///
    /// The dispatch must satisfy `Device::launch`'s contract after `Batch` validation.
    unsafe fn launch(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        pipeline: &Pipeline,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
        words: &[Word<'_>],
        has_previous_dispatch: bool,
    ) -> Result<()> {
        // SAFETY: forwarded from the caller.
        unsafe {
            match self {
                Self::Classic(batch) => batch.launch(
                    device,
                    pipeline,
                    config,
                    arguments,
                    words,
                    has_previous_dispatch,
                ),
                Self::Metal4(batch) => batch.launch(
                    device,
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

/// Kernel commands encoded by `Device::submit` into one ordered GPU batch.
///
/// A batch holds native command encoders, so it stays on the submitting thread:
///
/// ```compile_fail
/// use metal_oxide::Device;
/// fn check(device: &Device) {
///     // SAFETY: the batch encodes no kernel.
///     let submission = unsafe {
///         device.submit(|batch| {
///             std::thread::scope(|scope| {
///                 scope.spawn(|| drop(batch));
///             });
///             Ok(())
///         })
///     };
///     submission.unwrap().wait().unwrap();
/// }
/// ```
pub struct Batch<'d> {
    device: &'d Device,
    native: NativeBatch<'d>,
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

impl<'d> Batch<'d> {
    pub(super) fn new(device: &'d Device) -> Result<Self> {
        Ok(Self {
            device,
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

    /// Encodes a kernel after preceding kernels in this batch.
    ///
    /// # Safety
    ///
    /// The arguments, bounds, access modes, and participation rules must satisfy
    /// `Device::launch`'s contract. No external accesses may conflict with this batch.
    pub unsafe fn launch(
        &mut self,
        pipeline: &Pipeline,
        config: impl Into<DynamicLaunchConfig>,
        arguments: &[Argument<'_>],
    ) -> Result<()> {
        let config = config.into();
        if pipeline.device_id != self.device.id()
            || arguments
                .iter()
                .any(|a| a.device_id().is_some_and(|id| id != self.device.id()))
        {
            return Err(Error::DeviceMismatch);
        }
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
            pipeline.raw.staticThreadgroupMemoryLength(),
            |total, bytes| total.checked_add(bytes),
        );
        if shared_bytes.is_none_or(|bytes| bytes > self.device.max_threadgroup_memory_per_block()) {
            return Err(Error::InvalidLaunch(
                "kernel exceeds threadgroup memory capacity",
            ));
        }
        pipeline.validate_arguments(config.block, arguments)?;
        config.validate(
            pipeline.max_threads_per_block(),
            self.device.max_block_dimensions(),
        )?;
        if config.is_empty() {
            return Ok(());
        }
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
        let assignment = match &pipeline.checks {
            Some(kernel) => Some((
                kernel.gate_binding as usize,
                self.failures.assign(self.device.raw(), kernel.fallible)?,
            )),
            None => None,
        };
        if let Some((_, assignment)) = assignment.as_ref().filter(|(_, a)| a.snapshot) {
            let snapshot = self.device.snapshot()?;
            let words = [
                Word::new(&self.failures, 0, self.failures.failure_slot()),
                Word::new(&self.failures, 1, assignment.gate),
            ];
            self.pipelines.push(snapshot.raw.clone());
            let previous = std::mem::replace(&mut self.encoded, true);
            // SAFETY: one thread copies a word between live, retained failure-storage slots.
            unsafe {
                self.native.launch(
                    self.device.raw(),
                    snapshot,
                    DynamicLaunchConfig::new(Dim3::x(1), Dim3::x(1)),
                    &[],
                    &words,
                    previous,
                )?
            };
        }
        let words = assignment
            .iter()
            .flat_map(|(binding, assignment)| {
                std::iter::once((*binding, assignment.gate))
                    .chain(assignment.failure.map(|failure| (binding + 1, failure)))
            })
            .map(|(index, slot)| Word::new(&self.failures, index, slot))
            .collect::<Vec<_>>();
        self.pipelines.push(pipeline.raw.clone());
        let previous = std::mem::replace(&mut self.encoded, true);
        // SAFETY: validation passed; the submit caller provides the shader safety contract.
        unsafe {
            self.native.launch(
                self.device.raw(),
                pipeline,
                config,
                arguments,
                &words,
                previous,
            )?
        };
        if let (Some(kernel), Some((_, assignment))) = (&pipeline.checks, &assignment)
            && kernel.fallible
        {
            self.failures
                .record(kernel, self.dispatches, assignment.gate);
        }
        self.dispatches += 1;
        Ok(())
    }

    pub(super) fn commit<'a>(self) -> Submission<'a> {
        let completion = Arc::new(Completion::default());
        if self.dispatches == 0 {
            completion.finish(Ok(SubmissionReport::default()));
        } else {
            for resource in &self.resources {
                resource.access.register(&completion);
                if resource.written {
                    resource.access.mark_written();
                }
            }
            match self.native {
                NativeBatch::Classic(batch) => {
                    batch.commit(Arc::clone(&completion), self.pipelines, self.failures)
                }
                NativeBatch::Metal4(batch) => {
                    batch.commit(Arc::clone(&completion), self.pipelines, self.failures)
                }
            }
        }
        Submission {
            completion,
            marker: PhantomData,
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
