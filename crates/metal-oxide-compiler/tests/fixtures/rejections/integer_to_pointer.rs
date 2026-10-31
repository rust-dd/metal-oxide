#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let pointer = value as usize as *const u32;
    unsafe { output.store_unchecked(0, pointer.is_null() as u32) };
}
