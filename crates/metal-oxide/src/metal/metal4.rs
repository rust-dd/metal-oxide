use std::{ptr::NonNull, sync::Arc};

use block2::RcBlock;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions,
    MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLBuffer, MTLComputePipelineState,
    MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLStages,
};

use super::{Argument, Pipeline, batch::metal_size, completion::Completion};
use crate::{DynamicLaunchConfig, Error, Result};

pub(super) struct Metal4Batch {
    command: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Retained<ProtocolObject<dyn MTL4CommandAllocator>>,
    encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    bindings: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl Metal4Batch {
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    ) -> Result<Self> {
        let allocator = device
            .newCommandAllocator()
            .ok_or_else(|| Error::Command("could not create Metal 4 command allocator".into()))?;
        let residency = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|e| Error::Command(e.localizedDescription().to_string()))?;
        let command = device
            .newCommandBuffer()
            .ok_or_else(|| Error::Command("could not create Metal 4 command buffer".into()))?;
        command.beginCommandBufferWithAllocator(&allocator);
        let Some(encoder) = command.computeCommandEncoder() else {
            command.endCommandBuffer();
            return Err(Error::Command(
                "could not create Metal 4 compute encoder".into(),
            ));
        };
        encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::Device,
        );
        Ok(Self {
            command,
            allocator,
            encoder: Some(encoder),
            queue: queue.clone(),
            residency,
            tables: Vec::new(),
            bindings: Vec::new(),
        })
    }

    pub(super) unsafe fn launch(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        pipeline: &Pipeline,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
        has_previous_dispatch: bool,
    ) -> Result<()> {
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(arguments.len());
        descriptor.setInitializeBindings(true);
        let table = device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Command(e.localizedDescription().to_string()))?;
        for (index, argument) in arguments.iter().enumerate() {
            let buffer = argument.metal_buffer(device)?;
            self.residency
                .addAllocation(ProtocolObject::from_ref(&*buffer));
            // SAFETY: the table has this slot; the retained buffer stays resident through completion.
            unsafe { table.setAddress_atIndex(buffer.gpuAddress(), index) };
            self.bindings.push(buffer);
        }
        self.residency
            .addAllocation(ProtocolObject::from_ref(&*pipeline.raw));
        let encoder = self.encoder.as_ref().unwrap();
        if has_previous_dispatch {
            encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
        }
        encoder.setComputePipelineState(&pipeline.raw);
        encoder.setArgumentTable(Some(&table));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            metal_size(config.grid),
            metal_size(config.block),
        );
        self.tables.push(table);
        Ok(())
    }

    pub(super) fn commit(
        mut self,
        completion: Arc<Completion>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    ) {
        self.encoder.take().unwrap().endEncoding();
        self.residency.commit();
        self.command.useResidencySet(&self.residency);
        self.command.endCommandBuffer();
        let keepalive = (
            self.command.clone(),
            self.allocator.clone(),
            self.queue.clone(),
            self.residency.clone(),
            std::mem::take(&mut self.tables),
            std::mem::take(&mut self.bindings),
            pipelines,
        );
        let handler = RcBlock::new(
            move |feedback: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
                let _keepalive = &keepalive;
                let result = autoreleasepool(|_| {
                    // SAFETY: Metal passes live feedback after this commit's GPU workload completes.
                    unsafe { feedback.as_ref() }
                        .error()
                        .map_or(Ok(()), |e| Err(e.localizedDescription().to_string()))
                });
                completion.finish(result);
            },
        );
        let options = MTL4CommitOptions::new();
        let mut command = NonNull::from(&*self.command);
        // SAFETY: Metal copies the callback; the one-element command array is valid for this call.
        // The callback retains all immutable native state until completion, independently of the future.
        unsafe {
            options.addFeedbackHandler(RcBlock::as_ptr(&handler));
            self.queue
                .commit_count_options(NonNull::from(&mut command), 1, &options);
        }
    }
}

impl Drop for Metal4Batch {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
            self.command.endCommandBuffer();
        }
    }
}
