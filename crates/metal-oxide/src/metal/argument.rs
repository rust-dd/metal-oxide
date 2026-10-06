use std::{marker::PhantomData, ptr::NonNull, rc::Rc};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions};

use crate::{Error, GpuAtomic, GpuScalar, Result};
use metal_oxide_artifact::{Access, ParameterType, Scalar};

use super::{Buffer, buffer::Resource, completion::AccessState};

enum Value<'a> {
    Buffer {
        raw: &'a Retained<ProtocolObject<dyn MTLBuffer>>,
        state: &'a Rc<AccessState>,
        device_id: u64,
        ty: ParameterType,
    },
    F32(f32),
    I32(i32),
    U32(u32),
    U8(u8),
    U16(u16),
}

/// A borrowed resource or copied scalar, bound in slice order to Metal buffer slots.
pub struct Argument<'a> {
    value: Value<'a>,
    marker: PhantomData<&'a mut Rc<()>>,
}

impl<'a> Argument<'a> {
    /// Borrows an integer buffer exclusively for atomic shader accesses.
    pub fn atomic<T: GpuAtomic>(buffer: &'a mut Buffer<T>) -> Self {
        Self {
            value: Value::Buffer {
                raw: &buffer.raw,
                state: &buffer.access,
                device_id: buffer.device_id,
                ty: ParameterType::Buffer {
                    element: T::TYPE,
                    access: Access::Atomic,
                },
            },
            marker: PhantomData,
        }
    }
    /// The kernel must treat this buffer as read-only.
    pub fn read<T: GpuScalar>(buffer: &'a Buffer<T>) -> Self {
        Self {
            value: Value::Buffer {
                raw: &buffer.raw,
                state: &buffer.access,
                device_id: buffer.device_id,
                ty: ParameterType::Buffer {
                    element: T::TYPE,
                    access: Access::Read,
                },
            },
            marker: PhantomData,
        }
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
    pub fn write<T: GpuScalar>(buffer: &'a mut Buffer<T>) -> Self {
        Self {
            value: Value::Buffer {
                raw: &buffer.raw,
                state: &buffer.access,
                device_id: buffer.device_id,
                ty: ParameterType::Buffer {
                    element: T::TYPE,
                    access: Access::Write,
                },
            },
            marker: PhantomData,
        }
    }

    pub fn f32(value: f32) -> Self {
        Self {
            value: Value::F32(value),
            marker: PhantomData,
        }
    }

    pub fn i32(value: i32) -> Self {
        Self {
            value: Value::I32(value),
            marker: PhantomData,
        }
    }

    pub fn u32(value: u32) -> Self {
        Self {
            value: Value::U32(value),
            marker: PhantomData,
        }
    }

    pub fn u8(value: u8) -> Self {
        Self {
            value: Value::U8(value),
            marker: PhantomData,
        }
    }
    pub fn u16(value: u16) -> Self {
        Self {
            value: Value::U16(value),
            marker: PhantomData,
        }
    }

    pub(super) fn device_id(&self) -> Option<u64> {
        match self.value {
            Value::Buffer { device_id, .. } => Some(device_id),
            _ => None,
        }
    }

    pub(super) fn resource(&self) -> Option<Resource> {
        match &self.value {
            Value::Buffer { raw, state, .. } => Some(Resource {
                raw: (*raw).clone(),
                access: Rc::clone(state),
            }),
            _ => None,
        }
    }

    pub(super) fn metal4_buffer(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>> {
        let (source, len) = match &self.value {
            Value::Buffer { raw, .. } => return Ok((*raw).clone()),
            Value::F32(value) => (NonNull::from(value).cast::<u8>(), 4),
            Value::I32(value) => (NonNull::from(value).cast::<u8>(), 4),
            Value::U32(value) => (NonNull::from(value).cast::<u8>(), 4),
            Value::U8(value) => (NonNull::from(value), 1),
            Value::U16(value) => (NonNull::from(value).cast::<u8>(), 2),
        };
        let buffer = device
            .newBufferWithLength_options(16, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::AllocationFailed { bytes: 16 })?;
        // SAFETY: the scalar source is live and has len bytes; fresh shared storage owns 16 writable bytes.
        unsafe {
            let destination = buffer.contents().cast::<u8>().as_ptr();
            destination.write_bytes(0, 16);
            std::ptr::copy_nonoverlapping(source.as_ptr(), destination, len);
        }
        Ok(buffer)
    }

    pub(super) fn ty(&self) -> ParameterType {
        match self.value {
            Value::Buffer { ty, .. } => ty,
            Value::F32(_) => ParameterType::Scalar {
                scalar: Scalar::F32,
            },
            Value::I32(_) => ParameterType::Scalar {
                scalar: Scalar::I32,
            },
            Value::U32(_) => ParameterType::Scalar {
                scalar: Scalar::U32,
            },
            Value::U8(_) => ParameterType::Scalar { scalar: Scalar::U8 },
            Value::U16(_) => ParameterType::Scalar {
                scalar: Scalar::U16,
            },
        }
    }

    pub(super) unsafe fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        index: usize,
    ) {
        match &self.value {
            Value::Buffer { raw, .. } => {
                // SAFETY: launch checks the slot and device; its caller guarantees shader access and bounds.
                unsafe { encoder.setBuffer_offset_atIndex(Some(raw), 0, index) };
            }
            Value::F32(value) => {
                // SAFETY: Metal copies the live scalar immediately; launch validates the binding index.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, index) };
            }
            Value::I32(value) => {
                // SAFETY: Metal copies the live scalar immediately; launch validates the binding index.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, index) };
            }
            Value::U32(value) => {
                // SAFETY: Metal copies the live scalar immediately; launch validates the binding index.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, index) };
            }
            Value::U8(value) => {
                // SAFETY: Metal copies the live scalar; launch validates its type and slot.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 1, index) };
            }
            Value::U16(value) => {
                // SAFETY: Metal copies the live scalar; launch validates its type and slot.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 2, index) };
            }
        }
    }
}
