use std::{
    marker::PhantomData,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{MTLCopyAllDevices, MTLCreateSystemDefaultDevice, MTLDevice};

use crate::{Dim3, DynamicLaunchConfig, Error, GpuValue, Result};

use super::{Argument, Batch, Buffer, Pipeline, Submission, backend::Backend};

static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(1);

/// A Metal device and an ordered command queue confined to one host thread.
pub struct Device {
    pub(super) raw: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(super) backend: Backend,
    id: u64,
    marker: PhantomData<Rc<()>>,
}

impl Device {
    pub fn system_default() -> Result<Self> {
        let raw = MTLCreateSystemDefaultDevice().ok_or(Error::DeviceUnavailable)?;
        Self::from_native(raw)
    }

    /// Opens an independent ordered context for each available device in native enumeration order.
    /// Returns an error if any device cannot be initialized. The order is not a persistent identity.
    pub fn all() -> Result<Vec<Self>> {
        MTLCopyAllDevices()
            .into_iter()
            .map(Self::from_native)
            .collect()
    }

    /// Retains a native device and creates a new independent ordered queue context.
    /// Existing native queues and their pending work are not imported.
    pub fn from_native(raw: Retained<ProtocolObject<dyn MTLDevice>>) -> Result<Self> {
        if !raw.hasUnifiedMemory() {
            return Err(Error::UnsupportedDevice(format!(
                "{} does not have unified memory",
                raw.name()
            )));
        }
        let backend = Backend::new(&raw)?;
        #[allow(deprecated, reason = "try_update is unavailable at the MSRV")]
        let id = NEXT_DEVICE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::Command("device identity space exhausted".into()))?;
        Ok(Self {
            raw,
            backend,
            id,
            marker: PhantomData,
        })
    }

    pub fn name(&self) -> String {
        autoreleasepool(|_| self.raw.name().to_string())
    }

    /// Borrows the native device for resource creation and external Metal integration.
    pub fn as_native(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.raw
    }

    /// The device's recommended working-set size in bytes, not currently free memory
    /// or a guarantee that an allocation of this size will succeed.
    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.raw.recommendedMaxWorkingSetSize()
    }

    pub fn buffer_from_slice<T: GpuValue>(&self, values: &[T]) -> Result<Buffer<T>> {
        let mut buffer = Buffer::zeroed(self, values.len())?;
        buffer.as_mut_slice().copy_from_slice(values);
        Ok(buffer)
    }

    pub fn buffer_zeroed<T: GpuValue>(&self, len: usize) -> Result<Buffer<T>> {
        Buffer::zeroed(self, len)
    }

    pub fn max_block_dimensions(&self) -> Dim3 {
        let limits = self.raw.maxThreadsPerThreadgroup();
        let axis = |value: usize| value.min(u32::MAX as usize) as u32;
        Dim3::new(axis(limits.width), axis(limits.height), axis(limits.depth))
    }

    /// Maximum combined static and dynamic threadgroup storage per block, in bytes.
    pub fn max_threadgroup_memory_per_block(&self) -> usize {
        self.raw.maxThreadgroupMemoryLength()
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
    /// CPU views acquired during encoding must end before committing GPU writes to them.
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
        unsafe { self.submit(|batch| batch.launch(pipeline, config, arguments))? }
            .wait()
            .map(|_| ())
    }
}
