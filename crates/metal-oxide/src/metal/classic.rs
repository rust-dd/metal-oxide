use std::{
    ptr::NonNull,
    sync::Arc,
    task::{Context, Poll, ready},
};

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchQueueGlobalPriority, GlobalQueueIdentifier};
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
    queue::{Finish, Signal},
    sync::{Mutex, lock},
};
use crate::{DynamicLaunchConfig, Error, Result};

pub(super) mod capacity;
use capacity::{Capacity, LIMIT, Permit};

pub(super) struct ClassicQueue {
    pub(super) raw: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    capacity: Arc<Capacity>,
}

impl ClassicQueue {
    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let raw = device
            .newCommandQueueWithMaxCommandBufferCount(LIMIT)
            .ok_or_else(|| Error::Command("could not create command queue".into()))?;
        Ok(Self {
            raw,
            capacity: Arc::default(),
        })
    }
}

pub(super) struct ClassicBatch {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    bindings: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    threadgroup_slots: usize,
    permit: Option<Permit>,
}

struct CompletedCommand(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

// SAFETY: created only in a completed handler; the waiter observes immutable command status
// and retains the native object until all completion handlers have returned.
unsafe impl Send for CompletedCommand {}

impl CompletedCommand {
    fn wait(self) {
        self.0.waitUntilCompleted();
    }
}

impl ClassicBatch {
    /// A command buffer that first waits for `waits`, with one compute encoder: serial when every
    /// dispatch waits for the previous one, which beats barriers in a concurrent encoder there.
    pub(super) fn poll_new(
        queue: &ClassicQueue,
        log: Option<&ProtocolObject<dyn MTLLogState>>,
        serial: bool,
        waits: &[Signal],
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self>> {
        let permit = ready!(queue.capacity.poll(cx));
        let descriptor = MTLCommandBufferDescriptor::new();
        descriptor.setLogState(log);
        let command = autoreleasepool(|_| queue.raw.commandBufferWithDescriptor(&descriptor))
            .ok_or_else(|| Error::Command("could not create command buffer".into()))?;
        for wait in waits {
            command.encodeWaitForEvent_value(ProtocolObject::from_ref(&*wait.event), wait.value);
        }
        let dispatch = if serial {
            MTLDispatchType::Serial
        } else {
            MTLDispatchType::Concurrent
        };
        let encoder = autoreleasepool(|_| command.computeCommandEncoderWithDispatchType(dispatch))
            .ok_or_else(|| Error::Command("could not create compute encoder".into()))?;
        Poll::Ready(Ok(Self {
            command,
            encoder: Some(encoder),
            queue: queue.raw.clone(),
            bindings: Vec::new(),
            threadgroup_slots: 0,
            permit: Some(permit),
        }))
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
                    unsafe { encoder.setBuffer_offset_atIndex(Some(&buffer.0), 0, buffer_slot) };
                    self.bindings.push(buffer.0.clone());
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

    /// Commits the command buffer, which sets `signal` when its work finishes.
    pub(super) fn commit(
        mut self,
        completion: Arc<Completion>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        failures: Arc<Failures>,
        signal: &Signal,
        finish: Finish,
    ) {
        self.encoder.take().unwrap().endEncoding();
        self.command
            .encodeSignalEvent_value(ProtocolObject::from_ref(&*signal.event), signal.value);
        let queue = self.queue.clone();
        let bindings = std::mem::take(&mut self.bindings);
        let permit = Mutex::new(self.permit.take());
        let handler = RcBlock::new(
            move |command: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let _keepalive = (&pipelines, &queue, &bindings);
                let result = autoreleasepool(|_| {
                    // SAFETY: Metal passes a live command buffer to its completion handler.
                    let command = unsafe { command.as_ref() };
                    let failed = failures.finish();
                    if command.status() == MTLCommandBufferStatus::Completed {
                        match failed {
                            Some(failed) => Err(CommandFailure::from(failed)),
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
                finish.run();
                let permit = lock(&permit).take().unwrap();
                // SAFETY: Metal keeps this completed command alive throughout its callback.
                let command =
                    CompletedCommand(unsafe { Retained::retain(command.as_ptr()).unwrap() });
                DispatchQueue::global_queue(GlobalQueueIdentifier::Priority(
                    DispatchQueueGlobalPriority::Default,
                ))
                .exec_async(move || {
                    command.wait();
                    drop(permit);
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

#[cfg(all(test, not(loom)))]
mod tests;
