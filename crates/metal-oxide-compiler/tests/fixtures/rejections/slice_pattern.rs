#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(input: ReadBuffer<u32>, output: WriteBuffer<u32>) {
    let first = match input.as_slice() {
        [first, ..] => *first,
        [] => 0,
    };
    unsafe { output.store_unchecked(0, first) };
}
