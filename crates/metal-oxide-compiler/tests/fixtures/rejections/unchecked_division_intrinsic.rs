#![no_std]
#![feature(core_intrinsics)]
use metal_oxide_device::{WriteBuffer, kernel};
#[kernel]
pub unsafe fn unchecked_division(out: WriteBuffer<u32>, value: u32, divisor: u32) {
    unsafe { out.store_unchecked(0, core::intrinsics::unchecked_div(value, divisor)); }
}
