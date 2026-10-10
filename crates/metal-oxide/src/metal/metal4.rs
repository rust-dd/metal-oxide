use std::{cell::RefCell, ptr::NonNull, sync::Arc};

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

use super::{
    Argument, Pipeline,
    batch::metal_size,
    completion::{Completion, SubmissionReport},
    constants::Constants,
    failure::CommandFailure,
};
use crate::{DynamicLaunchConfig, Error, Result};

type Allocator = Retained<ProtocolObject<dyn MTL4CommandAllocator>>;

#[derive(Default)]
pub(super) struct AllocatorCache(RefCell<Option<(Allocator, Arc<Completion>)>>);

impl AllocatorCache {
    fn take(&self) -> Option<Allocator> {
        let (allocator, completion) = self.0.borrow_mut().take()?;
        if !completion.is_finished() {
            return None;
        }
        // Commit feedback has confirmed that the GPU no longer reads these commands.
        allocator.reset();
        Some(allocator)
    }
}

pub(super) struct Metal4Batch<'a> {
    command: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Allocator,
    encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    tables: Vec<Retained<ProtocolObject<dyn MTL4ArgumentTable>>>,
    bindings: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    threadgroup_slots: usize,
    cache: &'a AllocatorCache,
}

impl<'a> Metal4Batch<'a> {
    pub(super) fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &Retained<ProtocolObject<dyn MTL4CommandQueue>>,
        cache: &'a AllocatorCache,
    ) -> Result<Self> {
        let allocator = cache
            .take()
            .or_else(|| device.newCommandAllocator())
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
            threadgroup_slots: 0,
            cache,
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
        descriptor.setMaxBufferBindCount(
            arguments
                .iter()
                .filter(|a| a.threadgroup_bytes().is_none())
                .count(),
        );
        descriptor.setInitializeBindings(true);
        let table = device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Command(e.localizedDescription().to_string()))?;
        let encoder = self.encoder.as_ref().unwrap();
        let constants = Constants::new(device, arguments)?;
        if let Some(constants) = &constants {
            self.residency
                .addAllocation(ProtocolObject::from_ref(&*constants.buffer));
            self.bindings.push(constants.buffer.clone());
        }
        let mut buffer_slot = 0;
        let mut threadgroup_slot = 0;
        for argument in arguments {
            if let Some(bytes) = argument.threadgroup_bytes() {
                // SAFETY: Batch validated the slot and combined per-block storage size.
                unsafe { encoder.setThreadgroupMemoryLength_atIndex(bytes, threadgroup_slot) };
                threadgroup_slot += 1;
                continue;
            }
            let address = if argument.value_bytes().is_some() {
                constants.as_ref().unwrap().address(buffer_slot)
            } else {
                let buffer = argument.metal_buffer(device)?;
                self.residency
                    .addAllocation(ProtocolObject::from_ref(&*buffer));
                let address = buffer.gpuAddress();
                self.bindings.push(buffer);
                address
            };
            // SAFETY: the table has this slot; the aligned, immutable bytes stay
            // resident and retained through completion, independently of later dispatches.
            unsafe { table.setAddress_atIndex(address, buffer_slot) };
            buffer_slot += 1;
        }
        for slot in threadgroup_slot..self.threadgroup_slots {
            // SAFETY: a previously validated slot is unused by the next dispatch.
            unsafe { encoder.setThreadgroupMemoryLength_atIndex(0, slot) };
        }
        self.threadgroup_slots = threadgroup_slot;
        self.residency
            .addAllocation(ProtocolObject::from_ref(&*pipeline.raw));
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
        self.cache
            .0
            .replace(Some((self.allocator.clone(), Arc::clone(&completion))));
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
                    let feedback = unsafe { feedback.as_ref() };
                    match feedback.error() {
                        Some(error) => Err(CommandFailure::from_native(&error)),
                        None => Ok(SubmissionReport::from_gpu_times(
                            feedback.GPUStartTime(),
                            feedback.GPUEndTime(),
                        )),
                    }
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

impl Drop for Metal4Batch<'_> {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
            self.command.endCommandBuffer();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Metal 4 device on Apple Silicon"]
    fn allocator_reuse_requires_completed_gpu_work() {
        let device = crate::Device::system_default().unwrap();
        if !super::super::backend::Backend::supports_metal4(&device.raw) {
            return;
        }
        let allocator = device.raw.newCommandAllocator().unwrap();
        let completion = Arc::new(Completion::default());
        let cache = AllocatorCache::default();
        cache
            .0
            .replace(Some((allocator.clone(), Arc::clone(&completion))));
        assert!(cache.take().is_none());
        completion.finish(Ok(SubmissionReport::default()));
        cache.0.replace(Some((allocator.clone(), completion)));
        let reused = cache.take().unwrap();
        assert!(std::ptr::eq(&*allocator, &*reused));
        assert!(cache.take().is_none());
    }
}
