use std::{ptr::NonNull, sync::Arc};

use block2::RcBlock;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
};

use super::{Argument, Pipeline, batch::metal_size, completion::Completion};
use crate::{DynamicLaunchConfig, Error, Result};

pub(super) struct ClassicBatch {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
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
        })
    }

    pub(super) unsafe fn launch(
        &self,
        pipeline: &Pipeline,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
        has_previous_dispatch: bool,
    ) {
        let encoder = self.encoder.as_ref().unwrap();
        if has_previous_dispatch {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encoder.setComputePipelineState(&pipeline.raw);
        for (index, argument) in arguments.iter().enumerate() {
            // SAFETY: Batch validated bindings; its caller guarantees the shader contract.
            unsafe { argument.encode(encoder, index) };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            metal_size(config.grid),
            metal_size(config.block),
        );
    }

    pub(super) fn commit(
        mut self,
        completion: Arc<Completion>,
        buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    ) {
        self.encoder.take().unwrap().endEncoding();
        let queue = self.queue.clone();
        let handler = RcBlock::new(
            move |command: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let _keepalive = (&buffers, &pipelines, &queue);
                let result = autoreleasepool(|_| {
                    // SAFETY: Metal passes a live command buffer to its completion handler.
                    let command = unsafe { command.as_ref() };
                    if command.status() == MTLCommandBufferStatus::Completed {
                        Ok(())
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
