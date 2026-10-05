use std::{marker::PhantomData, rc::Rc};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLSize,
};

use crate::{Dispatch1d, Error, GpuScalar, Result};

use super::{Argument, Buffer, Pipeline};

/// A Metal device and a synchronous command queue confined to one host thread.
pub struct Device {
    pub(super) raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    marker: PhantomData<Rc<()>>,
}

impl Device {
    pub fn system_default() -> Result<Self> {
        let raw = MTLCreateSystemDefaultDevice().ok_or(Error::DeviceUnavailable)?;
        if !raw.hasUnifiedMemory() {
            return Err(Error::UnsupportedDevice(raw.name().to_string()));
        }
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| Error::Command("could not create queue".into()))?;
        Ok(Self {
            raw,
            queue,
            marker: PhantomData,
        })
    }

    pub fn name(&self) -> String {
        autoreleasepool(|_| self.raw.name().to_string())
    }

    pub fn buffer_from_slice<T: GpuScalar>(&self, values: &[T]) -> Result<Buffer<T>> {
        let mut buffer = Buffer::zeroed(self, values.len())?;
        buffer.as_mut_slice().copy_from_slice(values);
        Ok(buffer)
    }

    pub fn buffer_zeroed<T: GpuScalar>(&self, len: usize) -> Result<Buffer<T>> {
        Buffer::zeroed(self, len)
    }

    pub(super) fn id(&self) -> u64 {
        self.raw.registryID()
    }

    /// Encodes the kernel, waits for completion, then checks command status.
    ///
    /// # Safety
    ///
    /// Arguments must match the kernel's types, slots, lengths, and access modes.
    /// The kernel must stay in bounds, respect read-only resources, have no data
    /// races, and leave valid values in every written element. Its grid must
    /// satisfy all participation and synchronization requirements. The caller
    /// must ensure no external GPU or CPU access conflicts with this submission.
    pub unsafe fn dispatch(
        &self,
        pipeline: &Pipeline,
        grid: Dispatch1d,
        arguments: &[Argument<'_>],
    ) -> Result<()> {
        if pipeline.device_id != self.id()
            || arguments
                .iter()
                .any(|a| a.device_id().is_some_and(|id| id != self.id()))
        {
            return Err(Error::DeviceMismatch);
        }
        if arguments.len() > 31 {
            return Err(Error::TooManyArguments(arguments.len()));
        }
        let group_width = grid.group_width(
            pipeline.thread_execution_width(),
            pipeline.max_threads_per_threadgroup(),
        )?;
        if grid.threads() == 0 {
            return Ok(());
        }
        autoreleasepool(|_| {
            let command = self
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Command("could not create command buffer".into()))?;
            let encoder = command
                .computeCommandEncoder()
                .ok_or_else(|| Error::Command("could not create compute encoder".into()))?;
            encoder.setComputePipelineState(&pipeline.raw);
            for (index, argument) in arguments.iter().enumerate() {
                // SAFETY: slots and device ownership were checked; the caller supplies the shader contract.
                unsafe { argument.encode(&encoder, index) };
            }
            encoder.dispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: grid.threads() as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: group_width,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                let reason = command
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_else(|| {
                        format!("unexpected command status: {:?}", command.status())
                    });
                return Err(Error::Command(reason));
            }
            Ok(())
        })
    }
}
