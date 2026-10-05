#![no_std]
use metal_oxide_device::{ReadBuffer, kernel};

pub trait Elements {
    type Scalar;
    type Float;
}
impl Elements for () {
    type Scalar = u32;
    type Float = f32;
}

#[kernel]
pub unsafe fn concrete(
    value: <() as Elements>::Scalar,
    input: ReadBuffer<<() as Elements>::Float>,
) {
    let _ = (value, input);
}
