use std::{ptr::NonNull, sync::Arc};

use block2::RcBlock;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
};

use super::{
    Argument, Pipeline,
    batch::metal_size,
    completion::{Completion, SubmissionReport},
};
use crate::{DynamicLaunchConfig, Error, Result};

pub(super) struct ClassicBatch {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    bindings: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl ClassicBatch {
    pub(super) fn new(queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>) -> Result<Self> {
        let command = queue
            .commandBuffer()
            .ok_or_else(|| Error::Command("could not create command buffer".into()))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Command("could not create compute encoder".into()))?;
        Ok(Self {
            command,
            encoder: Some(encoder),
            queue: queue.clone(),
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
        let encoder = self.encoder.as_ref().unwrap();
        if has_previous_dispatch {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encoder.setComputePipelineState(&pipeline.raw);
        for (index, argument) in arguments.iter().enumerate() {
            let buffer = argument.metal_buffer(device)?;
            // SAFETY: Batch validated bindings; native buffers are retained through completion.
            unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, index) };
            self.bindings.push(buffer);
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            metal_size(config.grid),
            metal_size(config.block),
        );
        Ok(())
    }

    pub(super) fn commit(
        mut self,
        completion: Arc<Completion>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    ) {
        self.encoder.take().unwrap().endEncoding();
        let queue = self.queue.clone();
        let bindings = std::mem::take(&mut self.bindings);
        let handler = RcBlock::new(
            move |command: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let _keepalive = (&pipelines, &queue, &bindings);
                let result = autoreleasepool(|_| {
                    // SAFETY: Metal passes a live command buffer to its completion handler.
                    let command = unsafe { command.as_ref() };
                    if command.status() == MTLCommandBufferStatus::Completed {
                        Ok(SubmissionReport::from_gpu_times(
                            command.GPUStartTime(),
                            command.GPUEndTime(),
                        ))
                    } else {
                        Err(command
                            .error()
                            .map(|e| e.localizedDescription().to_string())
                            .unwrap_or_else(|| {
                                format!("unexpected command status: {:?}", command.status())
                            }))
                    }
                });
                completion.finish(result);
            },
        );
        // SAFETY: Metal copies the block. Captured native resources are immutable after commit,
        // remain retained until GPU completion, and no host-thread Rc state crosses the callback.
        unsafe { self.command.addCompletedHandler(RcBlock::as_ptr(&handler)) };
        self.command.commit();
    }
}

impl Drop for ClassicBatch {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
    }
}
