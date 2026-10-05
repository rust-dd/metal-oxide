use std::{marker::PhantomData, rc::Rc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::{Error, GpuScalar, Result};

use super::Device;

/// Owned, initialized shared storage. CPU slices cannot outlive the buffer.
///
/// Buffers are neither clonable nor transferable between host threads.
///
/// ```compile_fail
/// use metal_oxide::Device;
/// let device = Device::system_default().unwrap();
/// let buffer = device.buffer_zeroed::<u32>(4).unwrap();
/// std::thread::spawn(move || drop(buffer));
/// ```
pub struct Buffer<T: GpuScalar> {
    pub(super) raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) device_id: u64,
    len: usize,
    marker: PhantomData<(T, Rc<()>)>,
}

impl<T: GpuScalar> Buffer<T> {
    pub(super) fn zeroed(device: &Device, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(Error::LengthOverflow)?;
        if bytes > isize::MAX as usize {
            return Err(Error::LengthOverflow);
        }
        // Metal needs nonempty storage even when the logical buffer is empty.
        let allocation_bytes = bytes.max(size_of::<T>());
        let maximum = device.raw.maxBufferLength();
        if allocation_bytes > maximum {
            return Err(Error::BufferTooLarge {
                bytes: allocation_bytes,
                maximum,
            });
        }
        let raw = device
            .raw
            .newBufferWithLength_options(allocation_bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::AllocationFailed {
                bytes: allocation_bytes,
            })?;
        // SAFETY: newly allocated shared storage is writable and all GpuScalar types admit zero bits.
        unsafe {
            raw.contents()
                .cast::<u8>()
                .as_ptr()
                .write_bytes(0, allocation_bytes)
        };
        Ok(Self {
            raw,
            device_id: device.id(),
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
        // SAFETY: storage is initialized and aligned for T; every GPU submission completes before returning.
        unsafe { std::slice::from_raw_parts(self.raw.contents().cast::<T>().as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the exclusive borrow prevents other CPU access and overlapping GPU argument borrows.
        unsafe {
            std::slice::from_raw_parts_mut(self.raw.contents().cast::<T>().as_ptr(), self.len)
        }
    }
}
