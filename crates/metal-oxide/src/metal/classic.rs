use std::{ptr::NonNull, sync::Arc};

use block2::RcBlock;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferDescriptor,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLogState,
};

use super::{
    Argument, Pipeline,
    batch::{Word, metal_size},
    completion::{Completion, SubmissionReport},
    failure::CommandFailure,
    failures::Failures,
};
use crate::{DynamicLaunchConfig, Error, Result};

pub(super) struct ClassicBatch {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    bindings: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    threadgroup_slots: usize,
}

impl ClassicBatch {
    pub(super) fn new(
        queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
        log: Option<&ProtocolObject<dyn MTLLogState>>,
    ) -> Result<Self> {
        let descriptor = MTLCommandBufferDescriptor::new();
        descriptor.setLogState(log);
        let command = queue
            .commandBufferWithDescriptor(&descriptor)
            .ok_or_else(|| Error::Command("could not create command buffer".into()))?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| Error::Command("could not create compute encoder".into()))?;
        Ok(Self {
            command,
            encoder: Some(encoder),
            queue: queue.clone(),
            bindings: Vec::new(),
            threadgroup_slots: 0,
        })
    }

    pub(super) unsafe fn launch(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        pipeline: &Pipeline,
        config: DynamicLaunchConfig,
        arguments: &[Argument<'_>],
        words: &[Word<'_>],
        has_previous_dispatch: bool,
    ) -> Result<()> {
        let encoder = self.encoder.as_ref().unwrap();
        if has_previous_dispatch {
            encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
        encoder.setComputePipelineState(&pipeline.raw);
        let mut buffer_slot = 0;
        let mut threadgroup_slot = 0;
        for argument in arguments {
            if let Some(bytes) = argument.threadgroup_bytes() {
                // SAFETY: Batch validated the slot and combined per-block storage size.
                unsafe { encoder.setThreadgroupMemoryLength_atIndex(bytes, threadgroup_slot) };
                threadgroup_slot += 1;
                continue;
            }
            if let Some(bytes) = argument.value_bytes().filter(|bytes| bytes.len() <= 4096) {
                // SAFETY: canonical value bytes are initialized and nonempty. Metal
                // copies at most 4 KiB before returning; later dispatches cannot mutate them.
                unsafe {
                    encoder.setBytes_length_atIndex(
                        NonNull::new(bytes.as_ptr().cast_mut()).unwrap().cast(),
                        bytes.len(),
                        buffer_slot,
                    )
                };
            } else {
                let buffer = argument.metal_buffer(device)?;
                // SAFETY: Batch validated bindings; buffers stay retained through completion.
                unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, buffer_slot) };
                self.bindings.push(buffer);
            }
            buffer_slot += 1;
        }
        for slot in threadgroup_slot..self.threadgroup_slots {
            // SAFETY: a previously validated slot is unused by the next dispatch.
            unsafe { encoder.setThreadgroupMemoryLength_atIndex(0, slot) };
        }
        self.threadgroup_slots = threadgroup_slot;
        for word in words {
            // SAFETY: the submission's `Failures` retains the word's buffer through completion.
            unsafe { encoder.setBuffer_offset_atIndex(Some(word.buffer), word.offset, word.index) };
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
        failures: Failures,
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
                        match failures.failure() {
                            Some(failure) => Err(CommandFailure::Kernel(Box::new(failure))),
                            None => Ok(SubmissionReport::from_gpu_times(
                                command.GPUStartTime(),
                                command.GPUEndTime(),
                            )),
                        }
                    } else {
                        Err(command
                            .error()
                            .map(|e| CommandFailure::from_native(&e))
                            .unwrap_or_else(|| {
                                CommandFailure::Other(format!(
                                    "unexpected command status: {:?}",
                                    command.status()
                                ))
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
