#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn mixed_space_references(input: ReadBuffer<u32>, out: WriteBuffer<u32>, value: u32) {
    let references = [&value, &input.as_slice()[0]];
    unsafe { out.store_unchecked(0, *references[(value & 1) as usize]) };
}
