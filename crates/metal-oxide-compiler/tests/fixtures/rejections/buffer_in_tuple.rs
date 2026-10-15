#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(input: ReadBuffer<f32>, output: WriteBuffer<f32>) {
    let pair = (input, 0u32);
    unsafe { output.store_unchecked(0, pair.0.load_unchecked(pair.1)) };
}
