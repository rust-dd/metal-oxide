use std::{marker::PhantomData, rc::Rc, sync::Arc};

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

enum NativeBatch {
    Classic(ClassicBatch),
    Metal4(Metal4Batch),
}
use crate::{Dim3, DynamicLaunchConfig, Error, Result};

/// Kernel commands encoded by `Device::submit` into one ordered GPU batch.
pub struct Batch<'d> {
    device: &'d Device,
    native: NativeBatch,
    resources: Vec<Resource>,
    pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    dispatches: usize,
}

impl<'d> Batch<'d> {
    pub(super) fn new(device: &'d Device) -> Result<Self> {
        Ok(Self {
            device,
            native: match &device.backend {
                Backend::Classic(queue) => NativeBatch::Classic(ClassicBatch::new(queue)?),
                Backend::Metal4 { queue, .. } => {
                    NativeBatch::Metal4(Metal4Batch::new(&device.raw, queue)?)
                }
            },
            resources: Vec::new(),
            pipelines: Vec::new(),
            dispatches: 0,
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
        if arguments.len() > 31 {
            return Err(Error::TooManyArguments(arguments.len()));
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
                    .find(|existing| Rc::ptr_eq(&existing.access, &resource.access))
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
                    &self.device.raw,
                    pipeline,
                    config,
                    arguments,
                    self.dispatches != 0,
                )?,
                NativeBatch::Metal4(batch) => batch.launch(
                    &self.device.raw,
                    pipeline,
                    config,
                    arguments,
                    self.dispatches != 0,
                )?,
            }
        };
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
