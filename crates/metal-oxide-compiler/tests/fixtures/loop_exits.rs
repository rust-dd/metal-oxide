#![no_std]

pub mod control_helpers;

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn bounded_search(input: ReadBuffer<u32>, output: WriteBuffer<u32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i >= n {
        return;
    }
    // SAFETY: the caller supplies n input/output elements and distinct output writes.
    unsafe {
        let limit = input.load_unchecked(i);
        output.store_unchecked(i, control_helpers::search(limit));
    }
}

#[kernel]
pub unsafe fn nested_exits(input: ReadBuffer<u32>, output: WriteBuffer<u32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: the caller supplies n input/output elements and distinct output writes.
        unsafe {
            let mode = input.load_unchecked(i);
            output.store_unchecked(i, control_helpers::nested(mode));
        }
    }
}
