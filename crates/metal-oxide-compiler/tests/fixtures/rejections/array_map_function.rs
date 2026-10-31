#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn mapped(out: WriteBuffer<u32>, value: u32) {
    let reversed = [value, 1].map(u32::reverse_bits);
    unsafe { out.store_unchecked(0, reversed[0]) };
}
