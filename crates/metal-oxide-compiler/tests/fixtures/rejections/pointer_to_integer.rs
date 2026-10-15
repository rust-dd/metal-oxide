#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let address = &value as *const u32 as usize;
    unsafe { output.store_unchecked(0, address as u32) };
}
