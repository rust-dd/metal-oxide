#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let pair = (value, value).clone();
    unsafe { output.store_unchecked(0, pair.0) };
}
