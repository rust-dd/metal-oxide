use std::{
    any::TypeId,
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    rc::Rc,
};

use metal_oxide_artifact::{Access, Layout};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::{Device, completion::AccessState};
use crate::{Error, GpuValue, Result};

mod native;
#[cfg(test)]
mod tests;

pub(super) struct Resource {
    pub(super) access: Rc<AccessState>,
    pub(super) written: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Location {
    Cpu,
    Gpu,
    Synced,
}

struct ReadbackGuard<'a>(&'a Cell<bool>);

impl Drop for ReadbackGuard<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

pub(super) trait BufferBinding {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>>;
    fn resource(&self) -> Resource;
    fn device_id(&self) -> u64;
    fn prepare(&self, access: Access);
}

/// Shared Metal storage with initialized CPU views.
///
/// Primitive scalar views use the allocation directly. Other `GpuValue` types
/// use a CPU mirror and their explicit canonical-layout codec.
///
/// CPU access waits for pending GPU work. Buffers cannot move between host threads.
///
/// ```compile_fail
/// use metal_oxide::Device;
/// let device = Device::system_default().unwrap();
/// let buffer = device.buffer_zeroed::<u32>(4).unwrap();
/// std::thread::spawn(move || drop(buffer));
/// ```
pub struct Buffer<T: GpuValue> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    device_id: u64,
    access: Rc<AccessState>,
    pub(super) layout: Layout,
    values: Option<UnsafeCell<Vec<T>>>,
    location: Cell<Location>,
    decoding: Cell<bool>,
    len: usize,
    marker: PhantomData<Rc<()>>,
}

impl<T: GpuValue> Buffer<T> {
    pub(super) fn zeroed(device: &Device, len: usize) -> Result<Self> {
        let layout = crate::value::layout::<T>()?;
        let bytes = Self::byte_len(len)?;
        let allocation_bytes = bytes.max(T::SIZE);
        let maximum = device.raw.maxBufferLength();
        if allocation_bytes > maximum {
            return Err(Error::BufferTooLarge {
                bytes: allocation_bytes,
                maximum,
            });
        }
        let values = Self::cpu_values(len, bytes)?;
        let raw = device
            .raw
            .newBufferWithLength_options(allocation_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::AllocationFailed {
                bytes: allocation_bytes,
            })?;
        assert!(raw.contents().cast::<T>().as_ptr().is_aligned() || values.is_some());
        // SAFETY: fresh shared storage owns allocation_bytes writable bytes.
        unsafe {
            raw.contents()
                .cast::<u8>()
                .as_ptr()
                .write_bytes(0, allocation_bytes)
        };
        Ok(Self {
            raw,
            device_id: device.id(),
            access: Rc::new(AccessState::default()),
            layout,
            values,
            location: Cell::new(Location::Cpu),
            decoding: Cell::new(false),
            len,
            marker: PhantomData,
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

    fn cpu_values(len: usize, bytes: usize) -> Result<Option<UnsafeCell<Vec<T>>>> {
        if native_layout::<T>() {
            return Ok(None);
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| Error::AllocationFailed { bytes })?;
        values.resize(len, T::zeroed());
        Ok(Some(UnsafeCell::new(values)))
    }

    pub fn as_slice(&self) -> &[T] {
        self.download();
        match &self.values {
            // SAFETY: download completed the preceding exclusive GPU write; shared
            // GPU arguments do not mutate the mirror or the native allocation.
            Some(values) => unsafe { &*values.get() },
            // SAFETY: native_layout admits only aligned, identical primitive representations
            // with all bit patterns valid. The allocation is initialized and idle.
            None => unsafe {
                std::slice::from_raw_parts(self.raw.contents().cast::<T>().as_ptr(), self.len)
            },
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.download();
        self.location.set(Location::Cpu);
        match &mut self.values {
            Some(values) => values.get_mut(),
            // SAFETY: native_layout proves representation validity; download waited for
            // every GPU access, and &mut self excludes all other CPU views.
            None => unsafe {
                std::slice::from_raw_parts_mut(self.raw.contents().cast::<T>().as_ptr(), self.len)
            },
        }
    }

    /// Encodes dirty CPU values into shared Metal storage before a later dispatch.
    /// Launches also perform this automatically. GPU-authored values are preserved.
    pub fn upload(&mut self) {
        self.prepare(Access::Read);
    }

    fn download(&self) {
        assert!(
            !self.decoding.get(),
            "buffer readback cannot reenter its decoder"
        );
        self.access.synchronize();
        let Some(values) = &self.values else {
            return;
        };
        if !self.access.take_written() && self.location.get() != Location::Gpu {
            return;
        }
        // A decoder panic must leave the GPU copy authoritative for the next readback.
        self.location.set(Location::Gpu);
        self.decoding.set(true);
        let _guard = ReadbackGuard(&self.decoding);
        // SAFETY: pending GPU work completed; the preceding write required an exclusive buffer
        // borrow. No CPU slice can exist while location is Gpu, and the guard rejects nested access.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                self.raw.contents().cast::<u8>().as_ptr(),
                self.len * T::SIZE,
            );
            for (value, encoded) in (&mut *values.get())
                .iter_mut()
                .zip(bytes.chunks_exact(T::SIZE))
            {
                *value = T::decode(encoded);
            }
        }
        self.location.set(Location::Synced);
    }
}

impl<T: GpuValue> BufferBinding for Buffer<T> {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.raw
    }
    fn device_id(&self) -> u64 {
        self.device_id
    }
    fn resource(&self) -> Resource {
        Resource {
            access: Rc::clone(&self.access),
            written: false,
        }
    }
    fn prepare(&self, access: Access) {
        let Some(values) = &self.values else {
            return;
        };
        if self.location.get() == Location::Cpu {
            // SAFETY: CPU mutation waits for pending work, and captured borrows prevent concurrent
            // mutation. Only the Metal bytes change; existing read-only CPU slices remain valid.
            unsafe {
                let bytes = std::slice::from_raw_parts_mut(
                    self.raw.contents().cast::<u8>().as_ptr(),
                    self.len * T::SIZE,
                );
                for (value, encoded) in (&*values.get()).iter().zip(bytes.chunks_exact_mut(T::SIZE))
                {
                    encoded.fill(0);
                    value.encode(encoded);
                }
            }
            self.location.set(Location::Synced);
        }
        if access != Access::Read {
            self.location.set(Location::Gpu);
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
