#![no_std]

use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn sin_cos_pairs(input: ReadBuffer<f32>, out: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        unsafe {
            let x = input.load_unchecked(i);
            let (sine, cosine) = x.sin_cos();
            out.store_unchecked(i * 4, sine);
            out.store_unchecked(i * 4 + 1, cosine);
            out.store_unchecked(i * 4 + 2, x.sin());
            out.store_unchecked(i * 4 + 3, x.cos());
        }
    }
}
