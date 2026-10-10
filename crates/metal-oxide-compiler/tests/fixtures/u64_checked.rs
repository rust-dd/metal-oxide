#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn checked(out: WriteBuffer<u64>, a: u64, b: u64) {
    unsafe { out.store_unchecked(0, a * b + 1) };
}
