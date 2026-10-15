#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let references = [&value; 2];
    unsafe { output.store_unchecked(0, *references[1]) };
}
