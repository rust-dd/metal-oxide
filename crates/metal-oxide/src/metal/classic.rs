use std::{ptr::NonNull, sync::Arc};

use block2::RcBlock;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandBufferDescriptor,
    MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLDispatchType, MTLLogState,
};

use super::{
    argument::Binding,
    batch::{Word, metal_size},
    completion::{Completion, gpu_time},
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
    /// A command buffer with one compute encoder: serial when every dispatch waits for the
    /// previous one, which beats barriers in a concurrent encoder for such chains.
    pub(super) fn new(
        queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
        log: Option<&ProtocolObject<dyn MTLLogState>>,
        serial: bool,
    ) -> Result<Self> {
        let descriptor = MTLCommandBufferDescriptor::new();
        descriptor.setLogState(log);
        let command = queue
            .commandBufferWithDescriptor(&descriptor)
            .ok_or_else(|| Error::Command("could not create command buffer".into()))?;
        let dispatch = if serial {
            MTLDispatchType::Serial
        } else {
            MTLDispatchType::Concurrent
        };
        let encoder = command
            .computeCommandEncoderWithDispatchType(dispatch)
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
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        config: DynamicLaunchConfig,
        bindings: &[Binding],
        store: &[u8],
        words: &[Word<'_>],
    ) -> Result<()> {
        let encoder = self.encoder.as_ref().unwrap();
        encoder.setComputePipelineState(pipeline);
        let mut buffer_slot = 0;
        let mut threadgroup_slot = 0;
        for binding in bindings {
            match binding {
                Binding::Threadgroup(bytes) => {
                    // SAFETY: Batch validated the slot and combined per-block storage size.
                    unsafe { encoder.setThreadgroupMemoryLength_atIndex(*bytes, threadgroup_slot) };
                    threadgroup_slot += 1;
                    continue;
                }
                Binding::Value { bytes, .. } if bytes.len() <= 4096 => {
                    let bytes = &store[bytes.clone()];
                    // SAFETY: canonical value bytes are initialized and nonempty. Metal
                    // copies at most 4 KiB before returning; later dispatches cannot mutate them.
                    unsafe {
                        encoder.setBytes_length_atIndex(
                            NonNull::new(bytes.as_ptr().cast_mut()).unwrap().cast(),
                            bytes.len(),
                            buffer_slot,
                        )
                    };
                }
                Binding::Value { bytes, .. } => {
                    let bytes = &store[bytes.clone()];
                    let buffer = super::constants::allocate(device, bytes.len())?;
                    // SAFETY: fresh shared storage covers the bytes and is not yet committed.
                    unsafe {
                        let destination = buffer.contents().cast::<u8>().as_ptr();
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
                    }
                    // SAFETY: the buffer stays retained through completion.
                    unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer), 0, buffer_slot) };
                    self.bindings.push(buffer);
                }
                Binding::Buffer(buffer) => {
                    // SAFETY: Batch validated bindings; buffers stay retained through completion.
                    unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, buffer_slot) };
                    self.bindings.push(buffer.clone());
                }
            }
            buffer_slot += 1;
        }
        for slot in threadgroup_slot..self.threadgroup_slots {
            // SAFETY: a previously validated slot is unused by the next dispatch.
            unsafe { encoder.setThreadgroupMemoryLength_atIndex(0, slot) };
        }
        self.threadgroup_slots = threadgroup_slot;
        for word in words {
            // SAFETY: the command buffer's `Failures` retains the word's buffer through completion.
            unsafe { encoder.setBuffer_offset_atIndex(Some(word.buffer), word.offset, word.index) };
        }
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            metal_size(config.grid),
            metal_size(config.block),
        );
        Ok(())
    }

    /// Orders the later dispatches after every earlier one and makes their writes visible.
    pub(super) fn barrier(&self) {
        self.encoder
            .as_ref()
            .unwrap()
            .memoryBarrierWithScope(MTLBarrierScope::Buffers);
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
                            None => Ok(gpu_time(command.GPUStartTime(), command.GPUEndTime())),
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
