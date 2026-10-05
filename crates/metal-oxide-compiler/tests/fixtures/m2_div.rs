#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn divide(out: WriteBuffer<u32>, n: u32) {
    unsafe { out.store_unchecked(0, 12 / n); }
}
