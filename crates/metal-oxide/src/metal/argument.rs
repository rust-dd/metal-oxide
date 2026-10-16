use std::{any::TypeId, cell::RefCell, collections::HashSet, ops::Range, sync::Arc};

use super::{
    Buffer,
    buffer::{BufferBinding, Resource, Storage},
};
use crate::{Error, GpuAtomic, GpuValue, Result};
use metal_oxide_artifact::{Access, Layout, ParameterType};

enum Value<'a> {
    Buffer(&'a dyn BufferBinding),
    Inline([u8; 16]),
    Encoded(Vec<u8>),
    Threadgroup(usize),
}

/// The Rust element type and binding kind a pipeline checks once per parameter.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Signature {
    element: TypeId,
    kind: Kind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Value,
    Buffer(Access),
    Threadgroup,
}

/// A kernel parameter in the form a batch records and the native encoders bind.
pub(super) enum Binding {
    Buffer(Storage),
    /// Canonical value bytes in the batch's byte store.
    Value {
        bytes: Range<usize>,
        alignment: usize,
    },
    Threadgroup(usize),
}

/// A borrowed buffer, owned value, or block-local allocation in kernel parameter order.
pub struct Argument<'a> {
    value: Value<'a>,
    signature: Signature,
    layout: fn() -> Result<Layout>,
    size: usize,
    alignment: usize,
}

/// Validation data retained after recording releases the argument's borrow.
pub(super) struct ArgumentInfo {
    signature: Signature,
    layout: fn() -> Result<Layout>,
    length: Option<usize>,
    integer: Option<u64>,
    threadgroup: Option<usize>,
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
    /// let values = output.read().wait();
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
        Self::new::<T>(Value::Buffer(buffer), Kind::Buffer(access))
    }

    fn new<T: GpuValue>(value: Value<'a>, kind: Kind) -> Self {
        Self {
            value,
            signature: Signature {
                element: TypeId::of::<T>(),
                kind,
            },
            layout: crate::value::layout::<T>,
            size: T::SIZE,
            alignment: T::ALIGNMENT,
        }
    }

    pub fn value<T: GpuValue>(value: T) -> Result<Self> {
        checked_codec::<T>()?;
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
        Ok(Self::new::<T>(value, Kind::Value))
    }

    /// Reserves uninitialized block-local storage for `elements` canonical values.
    /// The native byte count rounds up to a nonzero multiple of 16. Each argument
    /// uses a separate threadgroup slot, independently of buffer/value slots.
    pub fn threadgroup<T: GpuValue>(elements: usize) -> Result<Self> {
        checked_codec::<T>()?;
        let bytes = elements
            .checked_mul(T::SIZE)
            .and_then(|bytes| bytes.checked_add(15))
            .map(|bytes| (bytes & !15).max(16))
            .ok_or(Error::LengthOverflow)?;
        Ok(Self::new::<T>(Value::Threadgroup(bytes), Kind::Threadgroup))
    }

    pub(super) fn info(&self) -> ArgumentInfo {
        let integer = match &self.value {
            Value::Inline(bytes) if self.size <= 8 => {
                let mut wide = [0; 8];
                wide[..self.size].copy_from_slice(&bytes[..self.size]);
                Some(u64::from_le_bytes(wide))
            }
            _ => None,
        };
        ArgumentInfo {
            signature: self.signature,
            layout: self.layout,
            length: self.length(),
            integer,
            threadgroup: match self.value {
                Value::Threadgroup(bytes) => Some(bytes),
                _ => None,
            },
        }
    }

    /// The parameter as the native encoders bind it; value bytes are appended to `store`.
    pub(super) fn binding(&self, store: &mut Vec<u8>) -> Binding {
        let bytes = match &self.value {
            Value::Buffer(buffer) => return Binding::Buffer(Storage(buffer.raw().clone())),
            Value::Threadgroup(bytes) => return Binding::Threadgroup(*bytes),
            Value::Inline(bytes) => &bytes[..self.size],
            Value::Encoded(bytes) => bytes,
        };
        let start = store.len();
        store.extend_from_slice(bytes);
        Binding::Value {
            bytes: start..store.len(),
            alignment: self.alignment,
        }
    }

    /// The element count of a buffer argument.
    pub(super) fn length(&self) -> Option<usize> {
        match &self.value {
            Value::Buffer(buffer) => Some(buffer.len()),
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
        match (&self.value, self.signature.kind) {
            (Value::Buffer(buffer), Kind::Buffer(access)) => Some(Resource {
                access: Arc::clone(buffer.access()),
                read: access != Access::Write,
                written: access != Access::Read,
            }),
            _ => None,
        }
    }

    pub(super) fn prepare(&self) {
        if let (Value::Buffer(buffer), Kind::Buffer(access)) = (&self.value, self.signature.kind) {
            buffer.prepare(access);
        }
    }
}

impl ArgumentInfo {
    pub(super) fn signature(&self) -> Signature {
        self.signature
    }

    pub(super) fn length(&self) -> Option<usize> {
        self.length
    }

    pub(super) fn threadgroup_bytes(&self) -> Option<usize> {
        self.threadgroup
    }

    pub(super) fn integer(&self, scalar: metal_oxide_artifact::Scalar) -> Option<i128> {
        use metal_oxide_artifact::Scalar;
        let bits = self.integer?;
        Some(match scalar {
            Scalar::U8 | Scalar::U16 | Scalar::U32 | Scalar::U64 => i128::from(bits),
            Scalar::I8 => i128::from(bits as u8 as i8),
            Scalar::I16 => i128::from(bits as u16 as i16),
            Scalar::I32 => i128::from(bits as u32 as i32),
            Scalar::F32 | Scalar::F16 => return None,
        })
    }

    /// Builds the canonical parameter type; launches check it once per pipeline slot.
    pub(super) fn ty(&self) -> Result<ParameterType> {
        let layout = (self.layout)()?;
        Ok(match self.signature.kind {
            Kind::Value => ParameterType::Value { layout },
            Kind::Buffer(access) => ParameterType::Buffer {
                stride: layout.size,
                element: layout,
                access,
            },
            Kind::Threadgroup => ParameterType::Threadgroup {
                stride: layout.size,
                element: layout,
            },
        })
    }
}

mod sealed {
    pub trait Sealed {}
}

impl<T: GpuValue> sealed::Sealed for &Buffer<T> {}
impl<T: GpuValue> sealed::Sealed for &mut Buffer<T> {}
impl<T: GpuValue> sealed::Sealed for Buffer<T> {}
impl<T: GpuValue> sealed::Sealed for Arc<Buffer<T>> {}

/// A buffer a kernel only reads: borrowed, owned, or shared through an `Arc`.
pub trait ReadArgument<T: GpuValue>: sealed::Sealed {
    fn buffer(&self) -> &Buffer<T>;
}

/// A buffer a kernel writes: exclusively borrowed or owned. An `Arc` buffer is read-only
/// because a GPU write must exclude every CPU view of it, and so is a shared borrow:
///
/// ```compile_fail,E0277
/// use metal_oxide::{Buffer, WriteArgument};
/// fn written(buffer: impl WriteArgument<f32>) {}
/// fn check(buffer: &Buffer<f32>) {
///     written(buffer);
/// }
/// ```
pub trait WriteArgument<T: GpuValue>: sealed::Sealed {
    fn buffer_mut(&mut self) -> &mut Buffer<T>;
}

impl<T: GpuValue> ReadArgument<T> for &Buffer<T> {
    fn buffer(&self) -> &Buffer<T> {
        self
    }
}

impl<T: GpuValue> ReadArgument<T> for &mut Buffer<T> {
    fn buffer(&self) -> &Buffer<T> {
        self
    }
}

impl<T: GpuValue> ReadArgument<T> for Buffer<T> {
    fn buffer(&self) -> &Buffer<T> {
        self
    }
}

impl<T: GpuValue> ReadArgument<T> for Arc<Buffer<T>> {
    fn buffer(&self) -> &Buffer<T> {
        self
    }
}

impl<T: GpuValue> WriteArgument<T> for &mut Buffer<T> {
    fn buffer_mut(&mut self) -> &mut Buffer<T> {
        self
    }
}

impl<T: GpuValue> WriteArgument<T> for Buffer<T> {
    fn buffer_mut(&mut self) -> &mut Buffer<T> {
        self
    }
}

/// Checks a codec against its canonical layout once per thread and type.
fn checked_codec<T: GpuValue>() -> Result<()> {
    thread_local! {
        static CHECKED: RefCell<HashSet<TypeId>> = RefCell::new(HashSet::new());
    }
    let id = TypeId::of::<T>();
    if CHECKED.with_borrow(|checked| checked.contains(&id)) {
        return Ok(());
    }
    crate::value::layout::<T>()?;
    CHECKED.with_borrow_mut(|checked| checked.insert(id));
    Ok(())
}
