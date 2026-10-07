#![no_std]

pub mod control_helpers;

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn classify(input: ReadBuffer<i32>, output: WriteBuffer<u32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: the caller supplies n input/output elements and distinct output writes.
        unsafe {
            let value = input.load_unchecked(i);
            output.store_unchecked(i, control_helpers::classify(value));
        }
    }
}
