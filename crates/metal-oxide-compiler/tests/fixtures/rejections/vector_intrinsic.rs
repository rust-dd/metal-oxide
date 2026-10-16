#![no_std]
#![feature(core_intrinsics, portable_simd)]

use core::{intrinsics::simd, simd::Simd};
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn rejected(output: WriteBuffer<f32>, value: f32) {
    let lanes = Simd::<f32, 4>::splat(value);
    // SAFETY: the vector uses the intrinsic's required floating-point element type.
    let result = unsafe { simd::simd_fsqrt(lanes) };
    // SAFETY: the caller supplies one output element and launches one thread.
    unsafe { output.store_unchecked(0, result[0]) };
}
