#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn unchecked_casts(input: ReadBuffer<f32>, out_u32: WriteBuffer<u32>, out_i32: WriteBuffer<i32>) {
    // SAFETY: the caller supplies values whose truncation fits both targets.
    unsafe {
        let value = input.load_unchecked(0);
        out_u32.store_unchecked(0, value.to_int_unchecked::<u32>());
        out_i32.store_unchecked(0, (-value).to_int_unchecked::<i32>());
    }
}
