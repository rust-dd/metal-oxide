use std::{
    ptr::NonNull,
    sync::{Arc, Mutex, MutexGuard},
    task::{Context, Poll, ready},
};

use block2::RcBlock;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandAllocator, MTL4CommandBuffer,
    MTL4CommandBufferOptions, MTL4CommandEncoder, MTL4CommandQueue, MTL4CommitFeedback,
    MTL4CommitOptions, MTL4ComputeCommandEncoder, MTL4VisibilityOptions, MTLBuffer,
    MTLComputePipelineState, MTLDevice, MTLLogState, MTLResidencySet, MTLResidencySetDescriptor,
    MTLSharedEvent, MTLStages,
};

use super::{
    argument::Binding,
    batch::{Word, metal_size},
    completion::{Completion, gpu_time},
    constants::Constants,
    failure::CommandFailure,
    failures::Failures,
    queue::Finish,
};
use crate::{DynamicLaunchConfig, Error, Result};

type Allocator = Retained<ProtocolObject<dyn MTL4CommandAllocator>>;
const MAX_CACHED_CONSTANT_BYTES: usize = 4096;

#[derive(Clone)]
struct Table {
    raw: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    capacity: usize,
}

impl Table {
    fn new(device: &ProtocolObject<dyn MTLDevice>, capacity: usize) -> Result<Self> {
        let descriptor = MTL4ArgumentTableDescriptor::new();
        descriptor.setMaxBufferBindCount(capacity);
        descriptor.setInitializeBindings(true);
        let raw = device
            .newArgumentTableWithDescriptor_error(&descriptor)
            .map_err(|e| Error::Command(e.localizedDescription().to_string()))?;
        Ok(Self { raw, capacity })
    }
}

#[derive(Clone)]
struct CommandStorage {
    allocator: Allocator,
    command: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    constants: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    tables: Vec<Table>,
}

impl CommandStorage {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self> {
        let allocator = device
            .newCommandAllocator()
            .ok_or_else(|| Error::Command("could not create Metal 4 command allocator".into()))?;
        let residency = device
            .newResidencySetWithDescriptor_error(&MTLResidencySetDescriptor::new())
            .map_err(|e| Error::Command(e.localizedDescription().to_string()))?;
        let command = device
            .newCommandBuffer()
            .ok_or_else(|| Error::Command("could not create Metal 4 command buffer".into()))?;
        Ok(Self {
            allocator,
            command,
            residency,
            constants: Vec::new(),
            tables: Vec::new(),
        })
    }
}

// SAFETY: one batch owns the storage, or the cache parks it behind a mutex. Its native objects
// are used by one thread at a time, and the cache returns them only after GPU completion.
unsafe impl Send for CommandStorage {}

/// The committed batches whose storage a device keeps, like the command buffers a classic
/// queue keeps in flight. A fresh allocator costs a kernel call on its first encoder.
const MAX_BATCHES: usize = 64;

/// Command storage of committed batches, reused once their GPU work has completed.
#[derive(Default)]
pub(super) struct BatchCache(Mutex<Vec<(CommandStorage, Arc<Completion>)>>);

impl BatchCache {
    /// Reuses finished storage, allocates below capacity, or wakes when storage is available.
    fn poll_take(&self, cx: &mut Context<'_>) -> Poll<Option<CommandStorage>> {
        loop {
            let mut entries = self.entries();
            if let Some(index) = entries
                .iter()
                .position(|(_, completion)| completion.is_finished())
            {
                let (storage, _) = entries.remove(index);
                drop(entries);
                // Commit feedback has confirmed that the GPU no longer reads these commands.
                storage.allocator.reset();
                return Poll::Ready(Some(storage));
            }
            if entries.len() < MAX_BATCHES {
                return Poll::Ready(None);
            }
            if !entries
                .iter()
                .any(|(_, completion)| completion.poll_finished(cx).is_ready())
            {
                return Poll::Pending;
            }
        }
    }

    /// Keeps the storage of a committed batch.
    fn put(&self, storage: CommandStorage, completion: Arc<Completion>) {
        self.entries().push((storage, completion));
    }

    fn entries(&self) -> MutexGuard<'_, Vec<(CommandStorage, Arc<Completion>)>> {
        self.0.lock().unwrap_or_else(|error| error.into_inner())
    }
}

pub(super) struct Metal4Batch {
    command: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    allocator: Allocator,
    encoder: Option<Retained<ProtocolObject<dyn MTL4ComputeCommandEncoder>>>,
    queue: Retained<ProtocolObject<dyn MTL4CommandQueue>>,
    residency: Retained<ProtocolObject<dyn MTLResidencySet>>,
    tables: Vec<Table>,
    reusable_tables: Vec<Table>,
    bindings: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    constants: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    reusable_constants: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    threadgroup_slots: usize,
}

impl Metal4Batch {
    /// A command buffer that starts with a queue barrier when `ordered`, so it follows the
    /// conflicting work committed before it to the same queue.
    pub(super) fn poll_new(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &Retained<ProtocolObject<dyn MTL4CommandQueue>>,
        cache: &BatchCache,
        log: Option<&ProtocolObject<dyn MTLLogState>>,
        ordered: bool,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self>> {
        let CommandStorage {
            allocator,
            command,
            residency,
            constants: reusable_constants,
            tables: reusable_tables,
        } = match ready!(cache.poll_take(cx)) {
            Some(storage) => storage,
            None => CommandStorage::new(device)?,
        };
        let options = MTL4CommandBufferOptions::new();
        options.setLogState(log);
        command.beginCommandBufferWithAllocator_options(&allocator, &options);
        let Some(encoder) = command.computeCommandEncoder() else {
            command.endCommandBuffer();
            return Poll::Ready(Err(Error::Command(
                "could not create Metal 4 compute encoder".into(),
            )));
        };
        if ordered {
            encoder.barrierAfterQueueStages_beforeStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
        }
        Poll::Ready(Ok(Self {
            command,
            allocator,
            encoder: Some(encoder),
            queue: queue.clone(),
            residency,
            tables: Vec::new(),
            reusable_tables,
            bindings: Vec::new(),
            constants: Vec::new(),
            reusable_constants,
            threadgroup_slots: 0,
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
        let buffer_count = bindings
            .iter()
            .filter(|binding| !matches!(binding, Binding::Threadgroup(_)))
            .count();
        let bound = buffer_count + words.len();
        let table = match self
            .reusable_tables
            .pop()
            .filter(|table| table.capacity >= bound)
        {
            Some(table) => table,
            None => Table::new(device, bound)?,
        };
        for slot in bound..table.capacity {
            // SAFETY: these valid slots are unused by this dispatch.
            unsafe { table.raw.setAddress_atIndex(0, slot) };
        }
        let encoder = self.encoder.as_ref().unwrap();
        // SAFETY: the cache returns storage only after GPU completion; each dispatch takes
        // a distinct allocation and keeps its bytes unchanged through the next completion.
        let constants =
            unsafe { Constants::new(device, bindings, store, self.reusable_constants.pop())? };
        if let Some(constants) = &constants {
            self.residency
                .addAllocation(ProtocolObject::from_ref(&*constants.buffer));
            self.bindings.push(constants.buffer.clone());
            if constants.buffer.length() <= MAX_CACHED_CONSTANT_BYTES {
                self.constants.push(constants.buffer.clone());
            }
        }
        let mut buffer_slot = 0;
        let mut threadgroup_slot = 0;
        for binding in bindings {
            let address = match binding {
                Binding::Threadgroup(bytes) => {
                    // SAFETY: Batch validated the slot and combined per-block storage size.
                    unsafe { encoder.setThreadgroupMemoryLength_atIndex(*bytes, threadgroup_slot) };
                    threadgroup_slot += 1;
                    continue;
                }
                Binding::Value { .. } => constants.as_ref().unwrap().address(buffer_slot),
                Binding::Buffer(buffer) => {
                    self.residency
                        .addAllocation(ProtocolObject::from_ref(&*buffer.0));
                    self.bindings.push(buffer.0.clone());
                    buffer.0.gpuAddress()
                }
            };
            // SAFETY: the table has this slot; the aligned, immutable bytes stay
            // resident and retained through completion, independently of later dispatches.
            unsafe { table.raw.setAddress_atIndex(address, buffer_slot) };
            buffer_slot += 1;
        }
        for slot in threadgroup_slot..self.threadgroup_slots {
            // SAFETY: a previously validated slot is unused by the next dispatch.
            unsafe { encoder.setThreadgroupMemoryLength_atIndex(0, slot) };
        }
        self.threadgroup_slots = threadgroup_slot;
        for word in words {
            self.residency
                .addAllocation(ProtocolObject::from_ref(&**word.buffer));
            // SAFETY: the table has this slot; the word stays resident and retained by the
            // command buffer's `Failures` through completion.
            unsafe {
                table
                    .raw
                    .setAddress_atIndex(word.buffer.gpuAddress() + word.offset as u64, word.index)
            };
        }
        self.residency
            .addAllocation(ProtocolObject::from_ref(pipeline));
        encoder.setComputePipelineState(pipeline);
        encoder.setArgumentTable(Some(&table.raw));
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            metal_size(config.grid),
            metal_size(config.block),
        );
        self.tables.push(table);
        Ok(())
    }

    /// Orders the later dispatches after every earlier one and makes their writes visible.
    pub(super) fn barrier(&self) {
        self.encoder
            .as_ref()
            .unwrap()
            .barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
                MTLStages::Dispatch,
                MTLStages::Dispatch,
                MTL4VisibilityOptions::Device,
            );
    }

    /// Commits the command buffer after the queue has waited for each event value in `waits`.
    pub(super) fn commit(
        mut self,
        cache: &BatchCache,
        completion: Arc<Completion>,
        pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        failures: Arc<Failures>,
        waits: &[(&ProtocolObject<dyn MTLSharedEvent>, u64)],
        finish: Finish,
    ) {
        self.encoder.take().unwrap().endEncoding();
        self.residency.commit();
        self.command.useResidencySet(&self.residency);
        self.command.endCommandBuffer();
        let storage = CommandStorage {
            allocator: self.allocator.clone(),
            command: self.command.clone(),
            residency: self.residency.clone(),
            constants: std::mem::take(&mut self.constants),
            tables: self.tables.clone(),
        };
        let stored = Arc::clone(&completion);
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
                    let failed = failures.finish();
                    match feedback.error() {
                        Some(error) => Err(CommandFailure::from_native(&error)),
                        None => match failed {
                            Some(failed) => Err(CommandFailure::from(failed)),
                            None => Ok(gpu_time(feedback.GPUStartTime(), feedback.GPUEndTime())),
                        },
                    }
                });
                // The cache becomes reusable only after residency releases this batch's resources.
                keepalive.3.removeAllAllocations();
                keepalive.3.commit();
                finish.run();
                completion.finish(result);
            },
        );
        let options = MTL4CommitOptions::new();
        let mut command = NonNull::from(&*self.command);
        for (event, value) in waits {
            self.queue
                .waitForEvent_value(ProtocolObject::from_ref(*event), *value);
        }
        // SAFETY: Metal copies the callback; the one-element command array is valid for this call.
        // The callback retains all immutable native state until completion, independently of the future.
        unsafe {
            options.addFeedbackHandler(RcBlock::as_ptr(&handler));
            self.queue
                .commit_count_options(NonNull::from(&mut command), 1, &options);
        }
        // Only committed work is cached, so a waiting `take` always finishes.
        cache.put(storage, stored);
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

#[cfg(test)]
mod tests;
