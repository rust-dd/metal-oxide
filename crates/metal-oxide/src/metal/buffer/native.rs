use std::sync::Arc;

use metal_oxide_artifact::Access;
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{MTLBuffer, MTLDevice, MTLHazardTrackingMode, MTLResource, MTLStorageMode};

use super::{Buffer, BufferBinding, Location, Storage, native_layout};
use crate::{Device, Error, GpuValue, Result};

impl<T: GpuValue> Buffer<T> {
    /// Adopts initialized canonical GPU values from retained native storage.
    ///
    /// Requires shared, hazard-tracked storage on the same physical device, with
    /// room for `len` values (at least one element even when `len` is zero).
    /// Primitive views use the allocation directly; custom codecs use their CPU mirror.
    ///
    /// # Safety
    ///
    /// External GPU work must have completed. The first `len` canonical values,
    /// including padding, must be initialized if validation succeeds. Storage must
    /// remain allocated, non-purgeable and non-aliasable. External aliases must stay
    /// idle until all runtime CPU views and encoded/pending GPU accesses end.
    /// `into_native` performs the synchronized handoff back to native code;
    /// merely dropping a buffer does not complete pending commands. The runtime may
    /// retain, release and read immutable properties of `raw` on any thread.
    pub unsafe fn from_native(
        device: &Device,
        raw: Retained<ProtocolObject<dyn MTLBuffer>>,
        len: usize,
    ) -> Result<Self> {
        crate::value::layout::<T>()?;
        let bytes = Self::byte_len(len)?;
        let same_device =
            autoreleasepool(|_| raw.device().registryID() == device.raw().registryID());
        if !same_device {
            return Err(Error::DeviceMismatch);
        }
        if raw.storageMode() != MTLStorageMode::Shared {
            return Err(Error::InvalidBuffer(
                "native import requires shared storage",
            ));
        }
        if raw.hazardTrackingMode() == MTLHazardTrackingMode::Untracked {
            return Err(Error::InvalidBuffer(
                "native import requires hazard tracking",
            ));
        }
        if raw.length() < bytes.max(T::SIZE) {
            return Err(Error::InvalidBuffer(
                "native storage is shorter than the requested extent",
            ));
        }
        if !(raw.contents().as_ptr() as usize).is_multiple_of(T::ALIGNMENT)
            || (native_layout::<T>() && !raw.contents().cast::<T>().as_ptr().is_aligned())
        {
            return Err(Error::InvalidBuffer(
                "native storage does not satisfy element alignment",
            ));
        }
        Ok(Self {
            mirror: Self::mirror(len, bytes, Location::Gpu)?,
            storage: Storage(raw),
            device_id: device.id(),
            access: Arc::default(),
            len,
        })
    }

    /// Waits for pending GPU accesses, encodes dirty CPU values and transfers native ownership.
    /// Pending command errors remain available through their `Submission`.
    ///
    /// ```compile_fail
    /// use metal_oxide::Device;
    /// let device = Device::system_default().unwrap();
    /// let buffer = device.buffer_zeroed::<u32>(4).unwrap();
    /// let view = buffer.as_slice();
    /// let native = buffer.into_native();
    /// println!("{view:?}");
    /// ```
    pub fn into_native(self) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        self.access.synchronize();
        self.prepare(Access::Read);
        self.storage.0
    }
}
