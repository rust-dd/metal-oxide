use std::{marker::PhantomData, ptr::NonNull, rc::Rc};

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder};

use crate::GpuScalar;

use super::Buffer;

enum Value<'a> {
    Buffer {
        raw: &'a ProtocolObject<dyn MTLBuffer>,
        device_id: u64,
    },
    F32(f32),
    I32(i32),
    U32(u32),
}

/// A borrowed resource or copied scalar, bound in slice order to Metal buffer slots.
pub struct Argument<'a> {
    value: Value<'a>,
    marker: PhantomData<&'a mut Rc<()>>,
}

impl<'a> Argument<'a> {
    /// The kernel must treat this buffer as read-only.
    pub fn read<T: GpuScalar>(buffer: &'a Buffer<T>) -> Self {
        Self {
            value: Value::Buffer {
                raw: &buffer.raw,
                device_id: buffer.device_id,
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
                device_id: buffer.device_id,
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

    pub(super) fn device_id(&self) -> Option<u64> {
        match self.value {
            Value::Buffer { device_id, .. } => Some(device_id),
            _ => None,
        }
    }

    pub(super) unsafe fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        index: usize,
    ) {
        match &self.value {
            Value::Buffer { raw, .. } => {
                // SAFETY: dispatch checks the slot and device; its caller guarantees shader access and bounds.
                unsafe { encoder.setBuffer_offset_atIndex(Some(raw), 0, index) };
            }
            Value::F32(value) => {
                // SAFETY: Metal copies the live scalar immediately; dispatch validates the binding index.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, index) };
            }
            Value::I32(value) => {
                // SAFETY: Metal copies the live scalar immediately; dispatch validates the binding index.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, index) };
            }
            Value::U32(value) => {
                // SAFETY: Metal copies the live scalar immediately; dispatch validates the binding index.
                unsafe { encoder.setBytes_length_atIndex(NonNull::from(value).cast(), 4, index) };
            }
        }
    }
}
