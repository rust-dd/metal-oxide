#![no_std]
#![allow(unconditional_panic)]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let values = [value, value];
    unsafe { output.store_unchecked(0, values[2]) };
}
