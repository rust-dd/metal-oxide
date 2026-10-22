#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn power(out: WriteBuffer<u32>, value: u32, exponent: u32) {
    unsafe { out.store_unchecked(0, value.pow(exponent)) };
}
