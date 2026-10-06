use std::{
    marker::PhantomData,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

use crate::{Dim3, DynamicLaunchConfig, Error, GpuScalar, Result};

use super::{Argument, Batch, Buffer, Pipeline, Submission};

static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(1);

/// A Metal device and an ordered command queue confined to one host thread.
pub struct Device {
    pub(super) raw: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    id: u64,
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
        let id = NEXT_DEVICE_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::Command("device identity space exhausted".into()))?;
        Ok(Self {
            raw,
            queue,
            id,
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
        self.id
    }

    /// Encodes and commits an ordered batch without waiting for the GPU.
    ///
    /// Returning an encoding error discards the entire batch. Captured borrows
    /// remain attached to the submission; dropping it does not cancel GPU work.
    ///
    /// ```compile_fail
    /// use metal_oxide::{Argument, Device, Dim3, LaunchConfig, Pipeline};
    /// fn check(device: &Device, pipeline: &Pipeline) {
    ///     let mut output = device.buffer_zeroed::<u32>(32).unwrap();
    ///     let submission = unsafe { device.submit(|batch| {
    ///         batch.launch(pipeline, LaunchConfig::<32>::new(Dim3::x(1)),
    ///             &[Argument::write(&mut output)])
    ///     }) }.unwrap();
    ///     let values = output.as_slice();
    ///     submission.wait().unwrap();
    ///     println!("{}", values.len());
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Every encoded kernel must satisfy `Device::launch`'s safety contract.
    /// Accesses to resources across kernels must respect the encoded order.
    pub unsafe fn submit<'a>(
        &'a self,
        encode: impl FnOnce(&mut Batch<'_>) -> Result<()> + 'a,
    ) -> Result<Submission<'a>> {
        autoreleasepool(|_| {
            let mut batch = Batch::new(self)?;
            encode(&mut batch)?;
            Ok(batch.commit())
        })
    }

    /// Encodes the kernel, waits for completion, then checks command status.
    ///
    /// Accepts const-generic or dynamically sized launch configurations.
    ///
    /// # Safety
    ///
    /// Arguments must match the kernel's types, slots, lengths, and access modes.
    /// The kernel must stay in bounds, respect read-only resources, have no data
    /// races, and leave valid values in every written element. Its grid and block
    /// must satisfy all participation and synchronization requirements. The caller
    /// must ensure no external GPU or CPU access conflicts with this submission.
    pub unsafe fn launch(
        &self,
        pipeline: &Pipeline,
        config: impl Into<DynamicLaunchConfig>,
        arguments: &[Argument<'_>],
    ) -> Result<()> {
        // SAFETY: this method forwards the caller's kernel contract to the batch.
        unsafe { self.submit(|batch| batch.launch(pipeline, config, arguments))? }.wait()
    }
}
