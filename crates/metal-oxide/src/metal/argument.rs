use std::{marker::PhantomData, rc::Rc};

use metal_oxide_artifact::{Access, ParameterType};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::{
    Buffer,
    buffer::{BufferBinding, Resource},
};
use crate::{Error, GpuAtomic, GpuValue, Result};

enum Value<'a> {
    Buffer(&'a dyn BufferBinding),
    Encoded(Vec<u8>),
}

/// A borrowed buffer or an encoded owned value, bound in slice order to Metal slots.
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
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(T::SIZE)
            .map_err(|_| Error::AllocationFailed { bytes: T::SIZE })?;
        bytes.resize(T::SIZE, 0);
        value.encode(&mut bytes);
        Ok(Self {
            value: Value::Encoded(bytes),
            ty: ParameterType::Value { layout },
            marker: PhantomData,
        })
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
                        access: Access::Write | Access::Atomic,
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
        let bytes = match &self.value {
            Value::Buffer(buffer) => return Ok(buffer.raw().clone()),
            Value::Encoded(bytes) => bytes,
        };
        let length = bytes.len().max(16);
        let maximum = device.maxBufferLength();
        if length > maximum {
            return Err(Error::BufferTooLarge {
                bytes: length,
                maximum,
            });
        }
        let buffer = device
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::AllocationFailed { bytes: length })?;
        // SAFETY: bytes is live; fresh shared storage owns length bytes and is not yet submitted.
        unsafe {
            let destination = buffer.contents().cast::<u8>().as_ptr();
            destination.write_bytes(0, length);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
        Ok(buffer)
    }
}
