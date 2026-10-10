use std::{marker::PhantomData, rc::Rc};

use metal_oxide_artifact::{Access, ParameterType};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice};

use super::{
    Buffer,
    buffer::{BufferBinding, Resource},
};
use crate::{Error, GpuAtomic, GpuValue, Result};

enum Value<'a> {
    Buffer(&'a dyn BufferBinding),
    Inline([u8; 16]),
    Encoded(Vec<u8>),
    Threadgroup(usize),
}

/// A borrowed buffer, owned value, or block-local allocation in kernel parameter order.
pub struct Argument<'a> {
    value: Value<'a>,
    ty: ParameterType,
    marker: PhantomData<&'a mut Rc<()>>,
}

impl<'a> Argument<'a> {
    /// Borrows an integer buffer exclusively for atomic shader accesses.
    pub fn atomic<T: GpuAtomic>(buffer: &'a mut Buffer<T>) -> Self {
        Self::buffer(buffer, Access::Atomic)
    }

    /// The kernel must treat this buffer as read-only.
    pub fn read<T: GpuValue>(buffer: &'a Buffer<T>) -> Self {
        Self::buffer(buffer, Access::Read)
    }

    /// Holds the exclusive borrow until this argument is dropped.
    ///
    /// ```compile_fail
    /// use metal_oxide::{Argument, Device};
    /// let device = Device::system_default().unwrap();
    /// let mut output = device.buffer_zeroed::<f32>(4).unwrap();
    /// let arguments = [Argument::write(&mut output)];
    /// let values = output.as_slice();
    /// println!("{} {}", arguments.len(), values.len());
    /// ```
    pub fn write<T: GpuValue>(buffer: &'a mut Buffer<T>) -> Self {
        Self::buffer(buffer, Access::Write)
    }

    /// Borrows a buffer exclusively for both reads and writes by the kernel.
    ///
    /// ```compile_fail
    /// use metal_oxide::{Argument, Device};
    /// let device = Device::system_default().unwrap();
    /// let mut buffer = device.buffer_zeroed::<f32>(4).unwrap();
    /// let arguments = [Argument::read(&buffer), Argument::read_write(&mut buffer)];
    /// println!("{}", arguments.len());
    /// ```
    pub fn read_write<T: GpuValue>(buffer: &'a mut Buffer<T>) -> Self {
        Self::buffer(buffer, Access::ReadWrite)
    }

    fn buffer<T: GpuValue>(buffer: &'a Buffer<T>, access: Access) -> Self {
        Self {
            value: Value::Buffer(buffer),
            ty: ParameterType::Buffer {
                element: buffer.layout.clone(),
                stride: buffer.layout.size,
                access,
            },
            marker: PhantomData,
        }
    }

    pub fn value<T: GpuValue>(value: T) -> Result<Self> {
        let layout = crate::value::layout::<T>()?;
        let value = if T::SIZE <= 16 {
            let mut bytes = [0; 16];
            value.encode(&mut bytes[..T::SIZE]);
            Value::Inline(bytes)
        } else {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(T::SIZE)
                .map_err(|_| Error::AllocationFailed { bytes: T::SIZE })?;
            bytes.resize(T::SIZE, 0);
            value.encode(&mut bytes);
            Value::Encoded(bytes)
        };
        Ok(Self {
            value,
            ty: ParameterType::Value { layout },
            marker: PhantomData,
        })
    }

    /// Reserves uninitialized block-local storage for `elements` canonical values.
    /// The native byte count rounds up to a nonzero multiple of 16. Each argument
    /// uses a separate threadgroup slot, independently of buffer/value slots.
    pub fn threadgroup<T: GpuValue>(elements: usize) -> Result<Self> {
        let element = crate::value::layout::<T>()?;
        let bytes = elements
            .checked_mul(T::SIZE)
            .and_then(|bytes| bytes.checked_add(15))
            .map(|bytes| (bytes & !15).max(16))
            .ok_or(Error::LengthOverflow)?;
        Ok(Self {
            value: Value::Threadgroup(bytes),
            ty: ParameterType::Threadgroup {
                stride: element.size,
                element,
            },
            marker: PhantomData,
        })
    }

    pub(super) fn threadgroup_bytes(&self) -> Option<usize> {
        match self.value {
            Value::Threadgroup(bytes) => Some(bytes),
            _ => None,
        }
    }

    pub(super) fn value_bytes(&self) -> Option<&[u8]> {
        match &self.value {
            Value::Inline(bytes) => {
                let ParameterType::Value { layout } = &self.ty else {
                    unreachable!()
                };
                Some(&bytes[..layout.size as usize])
            }
            Value::Encoded(bytes) => Some(bytes),
            _ => None,
        }
    }

    pub(super) fn device_id(&self) -> Option<u64> {
        match &self.value {
            Value::Buffer(buffer) => Some(buffer.device_id()),
            _ => None,
        }
    }
    pub(super) fn resource(&self) -> Option<Resource> {
        match &self.value {
            Value::Buffer(buffer) => {
                let mut resource = buffer.resource();
                resource.written = matches!(
                    self.ty,
                    ParameterType::Buffer {
                        access: Access::Write | Access::ReadWrite | Access::Atomic,
                        ..
                    }
                );
                Some(resource)
            }
            _ => None,
        }
    }
    pub(super) fn ty(&self) -> &ParameterType {
        &self.ty
    }
    pub(super) fn prepare(&self) {
        if let Value::Buffer(buffer) = &self.value {
            let ParameterType::Buffer { access, .. } = self.ty else {
                unreachable!()
            };
            buffer.prepare(access);
        }
    }

    pub(super) fn metal_buffer(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        if let Value::Buffer(buffer) = &self.value {
            return Ok(buffer.raw().clone());
        }
        let bytes = self
            .value_bytes()
            .expect("threadgroup storage has no device buffer");
        let buffer = super::constants::allocate(device, bytes.len())?;
        // SAFETY: bytes is live; fresh shared storage covers it and is not yet submitted.
        unsafe {
            let destination = buffer.contents().cast::<u8>().as_ptr();
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
        Ok(buffer)
    }
}
