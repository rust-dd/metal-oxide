#![no_std]
#![feature(core_intrinsics)]
#![allow(internal_features)]
use metal_oxide_device::{WriteBuffer, kernel};

struct Plain(u32);

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    let value = core::intrinsics::discriminant_value(&Plain(1));
    unsafe { output.store_unchecked(0, value as u32) };
}
