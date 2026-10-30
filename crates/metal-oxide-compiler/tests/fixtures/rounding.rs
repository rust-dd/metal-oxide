#![no_std]

use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn rounding(input: ReadBuffer<f32>, out: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        unsafe {
            let x = input.load_unchecked(i);
            out.store_unchecked(3 * i, x.round());
            out.store_unchecked(3 * i + 1, x.ceil());
            out.store_unchecked(3 * i + 2, x.trunc());
        }
    }
}
