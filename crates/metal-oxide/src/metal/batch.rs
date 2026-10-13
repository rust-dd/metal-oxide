use std::{marker::PhantomData, sync::Arc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::{
    Argument, Device, Pipeline, Submission,
    backend::Backend,
    buffer::Resource,
    classic::ClassicBatch,
    completion::{Completion, SubmissionReport},
    metal4::Metal4Batch,
};
use crate::{Dim3, DynamicLaunchConfig, Error, Result};

enum NativeBatch<'a> {
    Classic(ClassicBatch),
    Metal4(Metal4Batch<'a>),
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
    has_dispatches: bool,
}

impl<'d> Batch<'d> {
    pub(super) fn new(device: &'d Device) -> Result<Self> {
        Ok(Self {
            device,
            native: match device.backend() {
                Backend::Classic(queue) => NativeBatch::Classic(ClassicBatch::new(queue)?),
                Backend::Metal4 { queue, cache, .. } => {
                    NativeBatch::Metal4(Metal4Batch::new(device.raw(), queue, cache)?)
                }
            },
            resources: Vec::new(),
            pipelines: Vec::new(),
            has_dispatches: false,
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
        self.pipelines.push(pipeline.raw.clone());
        // SAFETY: validation passed; the submit caller provides the shader safety contract.
        unsafe {
            match &mut self.native {
                NativeBatch::Classic(batch) => batch.launch(
                    self.device.raw(),
                    pipeline,
                    config,
                    arguments,
                    self.has_dispatches,
                )?,
                NativeBatch::Metal4(batch) => batch.launch(
                    self.device.raw(),
                    pipeline,
                    config,
                    arguments,
                    self.has_dispatches,
                )?,
            }
        };
        self.has_dispatches = true;
        Ok(())
    }

    pub(super) fn commit<'a>(self) -> Submission<'a> {
        let completion = Arc::new(Completion::default());
        if !self.has_dispatches {
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
                    batch.commit(Arc::clone(&completion), self.pipelines)
                }
                NativeBatch::Metal4(batch) => batch.commit(Arc::clone(&completion), self.pipelines),
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
