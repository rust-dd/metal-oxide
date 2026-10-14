#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn mixed(input: ReadBuffer<f32>, output: WriteBuffer<f32>, flag: u32) {
    let local = 1.0_f32;
    let input = input.as_slice();
    let value = if flag == 0 { &local } else { &input[0] };
    // SAFETY: deliberately rejected before emission.
    unsafe { output.store_unchecked(0, *value) };
}
