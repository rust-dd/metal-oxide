#![no_std]
use metal_oxide_device::{WriteBuffer, block_idx, kernel, simdgroup, thread_idx};
#[kernel(block = (256, 1, 1))]
pub unsafe fn broadcast(output: WriteBuffer<f32>) {
    let i = block_idx().x * 256 + thread_idx().x;
    // SAFETY: full blocks participate, lane zero is active, and output covers the complete grid.
    let value = unsafe { simdgroup::shuffle(i as f32, 0) };
    // SAFETY: each thread writes its own element in a fully sized buffer.
    unsafe { output.store_unchecked(i, value); }
}
