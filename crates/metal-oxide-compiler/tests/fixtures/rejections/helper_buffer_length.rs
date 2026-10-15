#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

fn length(buffer: ReadBuffer<u32>) -> usize {
    buffer.len()
}

#[kernel]
pub unsafe fn helper_buffer_by_value(input: ReadBuffer<u32>, out: WriteBuffer<u32>) {
    unsafe { out.store_unchecked(0, length(input) as u32) };
}
