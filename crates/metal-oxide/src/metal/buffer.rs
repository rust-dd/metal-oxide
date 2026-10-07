use std::{
    cell::{Cell, UnsafeCell},
    marker::PhantomData,
    rc::Rc,
};

use metal_oxide_artifact::{Access, Layout};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::{Device, completion::AccessState};
use crate::{Error, GpuValue, Result};

pub(super) struct Resource {
    pub(super) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) access: Rc<AccessState>,
    pub(super) written: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Location {
    Cpu,
    Gpu,
    Synced,
}

pub(super) trait BufferBinding {
    fn raw(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>>;
    fn resource(&self) -> Resource;
    fn device_id(&self) -> u64;
    fn prepare(&self, access: Access);
}

/// Owned CPU values and shared Metal storage with an explicit GPU representation.
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
    values: UnsafeCell<Vec<T>>,
    location: Cell<Location>,
    len: usize,
    marker: PhantomData<Rc<()>>,
}

impl<T: GpuValue> Buffer<T> {
    pub(super) fn zeroed(device: &Device, len: usize) -> Result<Self> {
        let layout = crate::value::layout::<T>()?;
        let bytes = len.checked_mul(T::SIZE).ok_or(Error::LengthOverflow)?;
        if bytes > isize::MAX as usize
            || len
                .checked_mul(size_of::<T>())
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(Error::LengthOverflow);
        }
        let allocation_bytes = bytes.max(T::SIZE);
        let maximum = device.raw.maxBufferLength();
        if allocation_bytes > maximum {
            return Err(Error::BufferTooLarge {
                bytes: allocation_bytes,
                maximum,
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|_| Error::AllocationFailed { bytes })?;
        values.resize(len, T::zeroed());
        let raw = device
            .raw
            .newBufferWithLength_options(allocation_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::AllocationFailed {
                bytes: allocation_bytes,
            })?;
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
            values: UnsafeCell::new(values),
            location: Cell::new(Location::Cpu),
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

    pub fn as_slice(&self) -> &[T] {
        self.download();
        // SAFETY: download modifies the CPU mirror only after an exclusive GPU write borrow ended.
        // Once a shared slice exists, read-only GPU arguments do not modify this mirror.
        unsafe { &*self.values.get() }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.download();
        self.location.set(Location::Cpu);
        self.values.get_mut()
    }

    fn download(&self) {
        self.access.synchronize();
        if !self.access.take_written() && self.location.get() != Location::Gpu {
            return;
        }
        // SAFETY: pending GPU work completed; the preceding write required an exclusive buffer
        // borrow. No CPU slice can exist while location is Gpu. Decoding constructs owned values.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                self.raw.contents().cast::<u8>().as_ptr(),
                self.len * T::SIZE,
            );
            for (value, encoded) in (&mut *self.values.get())
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
            raw: self.raw.clone(),
            access: Rc::clone(&self.access),
            written: false,
        }
    }
    fn prepare(&self, access: Access) {
        if self.location.get() == Location::Cpu {
            // SAFETY: CPU mutation waits for pending work, and captured borrows prevent concurrent
            // mutation. Only the Metal bytes change; existing read-only CPU slices remain valid.
            unsafe {
                let bytes = std::slice::from_raw_parts_mut(
                    self.raw.contents().cast::<u8>().as_ptr(),
                    self.len * T::SIZE,
                );
                for (value, encoded) in (&*self.values.get())
                    .iter()
                    .zip(bytes.chunks_exact_mut(T::SIZE))
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
