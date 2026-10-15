#![no_std]
use metal_oxide_device::{ReadBuffer, kernel};

#[kernel]
pub unsafe fn invalid(input: ReadBuffer<u32>) {
    let local = 5u32;
    let _same = &raw const local == input.as_slice().as_ptr();
}
