#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

/// Adds two buffers element by element.
///
/// # Safety
///
/// Buffers must contain at least n elements. Output must not overlap either
/// input. The launch must be one-dimensional, with one writer per output index.
#[kernel]
pub unsafe fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, out: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: i is below n and the caller guarantees buffer bounds and independent output.
        unsafe { out.store_unchecked(i, a.load_unchecked(i) + b.load_unchecked(i)) };
    }
}
