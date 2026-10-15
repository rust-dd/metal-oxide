#![no_std]
#![feature(portable_simd)]

use core::simd::f32x4;
use metal_oxide_device::{WriteBuffer, kernel};

const LANES: f32x4 = f32x4::from_array([1.0, 2.0, 3.0, 4.0]);

#[kernel]
pub unsafe fn constant_lanes(out: WriteBuffer<f32>, gain: f32) {
    let lanes = LANES;
    let _copy = lanes;
    unsafe { out.store_unchecked(0, gain) };
}
