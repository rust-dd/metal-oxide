#![no_std]

use metal_oxide_device::{intrinsics, kernel};

#[kernel]
pub unsafe fn invalid_signature(value: u32) {
    let _ = intrinsics::__metal_sqrt_f32(value);
}
