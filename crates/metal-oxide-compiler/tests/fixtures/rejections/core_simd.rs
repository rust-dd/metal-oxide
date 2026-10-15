#![no_std]
#![feature(portable_simd)]

use core::simd::f32x4;
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn scaled_lanes(out: WriteBuffer<f32>, gain: f32) {
    let lanes = f32x4::splat(gain);
    unsafe { out.store_unchecked(0, lanes[0]) };
}
