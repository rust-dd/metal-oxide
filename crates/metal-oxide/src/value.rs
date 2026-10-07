use metal_oxide_artifact::{Layout, Scalar};

use crate::{Error, Result};

/// Owned values encoded field by field into the canonical GPU layout.
///
/// `SIZE` and `ALIGNMENT` must match `layout()`. Encoding initializes every byte,
/// including padding; decoding ignores padding and constructs a valid Rust value.
/// Neither operation copies the Rust memory representation. The byte slices have
/// exactly `SIZE` bytes. Generated record bindings implement this trait.
///
/// ```compile_fail
/// use metal_oxide::GpuValue;
/// fn element<T: GpuValue>() {}
/// element::<bool>();
/// ```
pub trait GpuValue: Copy + 'static {
    const SIZE: usize;
    const ALIGNMENT: usize;
    fn layout() -> Result<Layout>;
    fn zeroed() -> Self;
    fn encode(self, bytes: &mut [u8]);
    fn decode(bytes: &[u8]) -> Self;
}

pub(crate) fn layout<T: GpuValue>() -> Result<Layout> {
    let layout = T::layout()?;
    layout.validate()?;
    if layout.size != T::SIZE as u64 || layout.alignment != T::ALIGNMENT as u64 {
        return Err(Error::Artifact(metal_oxide_artifact::Error(
            "codec size/alignment do not match the GPU layout".into(),
        )));
    }
    Ok(layout)
}

macro_rules! scalar {
    ($($ty:ty => $scalar:ident),* $(,)?) => {$(
        impl GpuValue for $ty {
            const SIZE: usize = size_of::<Self>();
            const ALIGNMENT: usize = Self::SIZE;
            fn layout() -> Result<Layout> { Ok(Layout::scalar(Scalar::$scalar)) }
            fn zeroed() -> Self { 0 as Self }
            fn encode(self, bytes: &mut [u8]) { bytes.copy_from_slice(&self.to_le_bytes()); }
            fn decode(bytes: &[u8]) -> Self { Self::from_le_bytes(bytes.try_into().expect("scalar byte length")) }
        }
    )*};
}
scalar!(f32 => F32, u32 => U32, i32 => I32, u8 => U8, u16 => U16, i8 => I8, i16 => I16);

impl<T: GpuValue, const N: usize> GpuValue for [T; N] {
    const SIZE: usize = T::SIZE.saturating_mul(N);
    const ALIGNMENT: usize = T::ALIGNMENT;
    fn layout() -> Result<Layout> {
        Ok(Layout::array(
            layout::<T>()?,
            u32::try_from(N).map_err(|_| Error::LengthOverflow)?,
        )?)
    }
    fn zeroed() -> Self {
        std::array::from_fn(|_| T::zeroed())
    }
    fn encode(self, bytes: &mut [u8]) {
        assert_eq!(bytes.len(), Self::SIZE);
        for (value, bytes) in self.into_iter().zip(bytes.chunks_exact_mut(T::SIZE)) {
            value.encode(bytes);
        }
    }
    fn decode(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), Self::SIZE);
        std::array::from_fn(|index| T::decode(&bytes[index * T::SIZE..(index + 1) * T::SIZE]))
    }
}

pub(crate) const fn aligned(value: usize, alignment: usize) -> usize {
    value.saturating_add(alignment - 1) & !(alignment - 1)
}

const fn tuple_size(fields: &[(usize, usize)]) -> usize {
    let mut size = 0;
    let mut alignment = 1;
    let mut index = 0;
    while index < fields.len() {
        let (bytes, align) = fields[index];
        size = aligned(size, align).saturating_add(bytes);
        if align > alignment {
            alignment = align;
        }
        index += 1;
    }
    aligned(size, alignment)
}

macro_rules! tuple {
    ($($ty:ident:$index:tt),+) => {
        impl<$($ty: GpuValue),+> GpuValue for ($($ty,)+) {
            const SIZE: usize = tuple_size(&[$(($ty::SIZE, $ty::ALIGNMENT)),+]);
            const ALIGNMENT: usize = {
                let mut alignment = 1;
                $(if $ty::ALIGNMENT > alignment { alignment = $ty::ALIGNMENT; })+
                alignment
            };
            fn layout() -> Result<Layout> { Ok(Layout::tuple(vec![$(layout::<$ty>()?),+])?) }
            fn zeroed() -> Self { ($($ty::zeroed(),)+) }
            fn encode(self, bytes: &mut [u8]) {
                assert_eq!(bytes.len(), Self::SIZE);
                bytes.fill(0);
                let mut offset = 0;
                $(offset = aligned(offset, $ty::ALIGNMENT);
                  self.$index.encode(&mut bytes[offset..offset + $ty::SIZE]);
                  offset += $ty::SIZE;)+
                let _ = offset;
            }
            fn decode(bytes: &[u8]) -> Self {
                assert_eq!(bytes.len(), Self::SIZE);
                let mut offset = 0;
                let value = ($({
                    offset = aligned(offset, $ty::ALIGNMENT);
                    let value = $ty::decode(&bytes[offset..offset + $ty::SIZE]);
                    offset += $ty::SIZE;
                    value
                },)+);
                let _ = offset;
                value
            }
        }
    };
}
tuple!(A:0);
tuple!(A:0, B:1);
tuple!(A:0, B:1, C:2);
tuple!(A:0, B:1, C:2, D:3);
tuple!(A:0, B:1, C:2, D:3, E:4);
tuple!(A:0, B:1, C:2, D:3, E:4, F:5);
tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
tuple!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);
