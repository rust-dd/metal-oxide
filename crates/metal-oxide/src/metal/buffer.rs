use std::{any::TypeId, ptr::NonNull, sync::Arc};

use metal_oxide_artifact::Access;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::{Device, access::AccessState};
use crate::{Error, GpuValue, Result};

mod mirror;
mod native;
#[cfg(test)]
mod tests;
mod views;

pub(super) use mirror::{Location, Mirror};
pub use views::{Read, Write};

/// A buffer argument's access state and how the kernel uses it.
pub(super) struct Resource {
    pub(super) access: Arc<AccessState>,
    pub(super) read: bool,
    pub(super) written: bool,
}

/// The native allocation behind a buffer.
#[derive(Clone)]
pub(super) struct Storage(pub(super) Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: the runtime retains, releases and reads immutable properties of the native buffer on
// any thread. Its bytes are only accessed under the `AccessState` and `Mirror` protocols.
unsafe impl Send for Storage {}
// SAFETY: see the `Send` implementation; shared access never mutates the native object.
unsafe impl Sync for Storage {}

impl Storage {
    fn contents(&self) -> NonNull<u8> {
        self.0.contents().cast()
    }
}

pub(super) trait BufferBinding: Sync {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>>;
    fn len(&self) -> usize;
    fn access(&self) -> &Arc<AccessState>;
    fn device_id(&self) -> u64;
    fn prepare(&self, access: Access);
}

/// Shared Metal storage with initialized CPU views; non-primitive values go through a CPU
/// mirror. CPU access waits for pending GPU work, and GPU writes need exclusive access.
pub struct Buffer<T: GpuValue> {
    storage: Storage,
    device_id: u64,
    access: Arc<AccessState>,
    mirror: Option<Mirror<T>>,
    len: usize,
}

impl<T: GpuValue> Buffer<T> {
    pub(super) fn zeroed(device: &Device, len: usize) -> Result<Self> {
        crate::value::layout::<T>()?;
        let bytes = Self::byte_len(len)?;
        let allocation_bytes = bytes.max(T::SIZE);
        let maximum = device.raw().maxBufferLength();
        if allocation_bytes > maximum {
            return Err(Error::BufferTooLarge {
                bytes: allocation_bytes,
                maximum,
            });
        }
        let mirror = Self::mirror(len, bytes, Location::Cpu)?;
        let raw = device
            .raw()
            .newBufferWithLength_options(allocation_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::AllocationFailed {
                bytes: allocation_bytes,
            })?;
        assert!(raw.contents().cast::<T>().as_ptr().is_aligned() || mirror.is_some());
        // SAFETY: fresh shared storage owns allocation_bytes writable bytes.
        unsafe {
            raw.contents()
                .cast::<u8>()
                .as_ptr()
                .write_bytes(0, allocation_bytes)
        };
        Ok(Self {
            storage: Storage(raw),
            device_id: device.id(),
            access: Arc::default(),
            mirror,
            len,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn byte_len(len: usize) -> Result<usize> {
        let bytes = len.checked_mul(T::SIZE).ok_or(Error::LengthOverflow)?;
        if bytes > isize::MAX as usize
            || len
                .checked_mul(size_of::<T>())
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(Error::LengthOverflow);
        }
        Ok(bytes)
    }

    fn mirror(len: usize, bytes: usize, location: Location) -> Result<Option<Mirror<T>>> {
        if native_layout::<T>() {
            return Ok(None);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| Error::AllocationFailed { bytes })?;
        values.resize(len, T::zeroed());
        Ok(Some(Mirror::new(values, location)))
    }

    /// The values, once pending GPU work has finished; awaiting polls GPU completion.
    ///
    /// # Panics
    ///
    /// Polling or waiting panics while an uncommitted batch has recorded a write to this buffer.
    pub fn read(&self) -> Read<'_, T> {
        Read::new(self)
    }

    /// The values for CPU writes, once pending GPU work has finished; the next dispatch that
    /// uses the buffer encodes them.
    ///
    /// # Panics
    ///
    /// Polling or waiting panics while an uncommitted batch has recorded this buffer.
    pub fn write(&mut self) -> Write<'_, T> {
        Write::new(self)
    }

    /// # Safety
    ///
    /// No GPU write to the buffer may be recorded or pending.
    unsafe fn values(&self) -> &[T] {
        match &self.mirror {
            // SAFETY: the storage holds len canonical values, and no GPU write is pending.
            Some(mirror) => unsafe { mirror.download(&self.access, self.storage.contents()) },
            // SAFETY: native_layout admits only aligned, identical primitive representations
            // with all bit patterns valid. The allocation is initialized, and GPU writes need `&mut`.
            None => unsafe {
                std::slice::from_raw_parts(self.storage.contents().cast::<T>().as_ptr(), self.len)
            },
        }
    }

    /// # Safety
    ///
    /// No GPU access to the buffer may be recorded or pending.
    unsafe fn values_mut(&mut self) -> &mut [T] {
        let bytes = self.storage.contents();
        match &mut self.mirror {
            // SAFETY: the storage holds len canonical values, and no GPU access is pending.
            Some(mirror) => unsafe { mirror.download_mut(&self.access, bytes) },
            // SAFETY: native_layout proves representation validity; every GPU access finished,
            // and &mut self excludes all other CPU views.
            None => unsafe { std::slice::from_raw_parts_mut(bytes.cast::<T>().as_ptr(), self.len) },
        }
    }

    pub(super) fn from_slice(device: &Device, values: &[T]) -> Result<Self> {
        let mut buffer = Self::zeroed(device, values.len())?;
        // SAFETY: the buffer was just created, so no GPU access exists.
        unsafe { buffer.values_mut() }.copy_from_slice(values);
        Ok(buffer)
    }

    /// Encodes dirty CPU values into shared Metal storage before a later dispatch.
    /// Launches also perform this automatically. GPU-authored values are preserved.
    ///
    /// # Panics
    ///
    /// Panics while an uncommitted batch has recorded this buffer.
    pub fn upload(&mut self) {
        self.access.assert_unrecorded();
        self.prepare(Access::Read);
    }
}

impl<T: GpuValue> BufferBinding for Buffer<T> {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.storage.0
    }
    fn len(&self) -> usize {
        self.len
    }
    fn device_id(&self) -> u64 {
        self.device_id
    }
    fn access(&self) -> &Arc<AccessState> {
        &self.access
    }
    fn prepare(&self, access: Access) {
        if let Some(mirror) = &self.mirror {
            // SAFETY: the storage holds len canonical values; GPU work sees them after encoding.
            unsafe { mirror.prepare(access, self.storage.contents()) };
        }
    }
}

fn native_layout<T: GpuValue>() -> bool {
    cfg!(target_endian = "little")
        && [
            TypeId::of::<f32>(),
            TypeId::of::<crate::F16>(),
            TypeId::of::<u32>(),
            TypeId::of::<u64>(),
            TypeId::of::<i32>(),
            TypeId::of::<u16>(),
            TypeId::of::<i16>(),
            TypeId::of::<u8>(),
            TypeId::of::<i8>(),
        ]
        .contains(&TypeId::of::<T>())
}
