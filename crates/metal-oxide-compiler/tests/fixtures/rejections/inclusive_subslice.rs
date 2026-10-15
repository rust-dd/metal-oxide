#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(input: ReadBuffer<u32>, output: WriteBuffer<u32>) {
    let head = &input.as_slice()[0..=1];
    unsafe { output.store_unchecked(0, head[0]) };
}
