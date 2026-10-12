#![no_std]

use metal_oxide_device::{WriteBuffer, kernel, thread_idx};

/// # Safety
/// `output` holds one value per thread.
#[kernel]
pub unsafe fn range_inclusive(output: WriteBuffer<u32>, n: u32) {
    let mut sum = 0;
    for k in 0..=n {
        sum += k;
    }
    unsafe { output.store_unchecked(thread_idx().x, sum) };
}
