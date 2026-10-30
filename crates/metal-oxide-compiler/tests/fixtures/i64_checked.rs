#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn checked(out: WriteBuffer<i64>, a: i64, b: i64) {
    unsafe { out.store_unchecked(0, a / b) };
    unsafe { out.store_unchecked(1, a * b + 1) };
}
