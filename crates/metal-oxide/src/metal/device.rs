use std::{marker::PhantomData, rc::Rc};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLSize,
};

use crate::{Dim3, Error, GpuScalar, LaunchConfig, Result};

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

    pub fn max_block_dimensions(&self) -> Dim3 {
        let limits = self.raw.maxThreadsPerThreadgroup();
        let axis = |value: usize| value.min(u32::MAX as usize) as u32;
        Dim3::new(axis(limits.width), axis(limits.height), axis(limits.depth))
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
    pub unsafe fn launch(
        &self,
        pipeline: &Pipeline,
        config: LaunchConfig,
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
        config.validate(
            pipeline.max_threads_per_block(),
            self.max_block_dimensions(),
        )?;
        if config.is_empty() {
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
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                metal_size(config.grid),
                metal_size(config.block),
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

fn metal_size(dimensions: Dim3) -> MTLSize {
    MTLSize {
        width: dimensions.x as usize,
        height: dimensions.y as usize,
        depth: dimensions.z as usize,
    }
}
