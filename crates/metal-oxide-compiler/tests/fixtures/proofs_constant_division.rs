#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn constant_dividend(out: WriteBuffer<i32>, divisor: i32) {
    unsafe { out.store_unchecked(0, i32::MIN / divisor) };
}

#[kernel]
pub unsafe fn constant_divisor(out: WriteBuffer<i8>, value: i8) {
    unsafe { out.store_unchecked(0, value % -1) };
}
