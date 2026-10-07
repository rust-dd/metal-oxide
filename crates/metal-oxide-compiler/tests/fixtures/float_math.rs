#![no_std]

use metal_oxide_device::{WriteBuffer, kernel, math};

#[kernel]
pub unsafe fn float_math(out: WriteBuffer<f32>, a: f32, b: f32, c: f32) {
    unsafe {
        out.store_unchecked(0, a.abs());
        out.store_unchecked(1, a.min(b));
        out.store_unchecked(2, a.max(b));
        out.store_unchecked(3, math::sqrt(a));
        out.store_unchecked(4, math::fma(a, b, c));
        out.store_unchecked(5, a * b + c);
        out.store_unchecked(6, f32::from_bits(a.to_bits()));
    }
}
