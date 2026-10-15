#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    let units = [(); 4];
    unsafe { output.store_unchecked(0, units.len() as u32) };
}
