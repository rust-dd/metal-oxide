#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn buffer_in_tuple_reference(input: ReadBuffer<u32>, out: WriteBuffer<u32>) {
    let pair = (&input, 1_u32);
    unsafe { out.store_unchecked(0, pair.0.len() as u32 + pair.1) };
}
