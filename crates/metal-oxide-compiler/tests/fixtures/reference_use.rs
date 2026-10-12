#![no_std]

use metal_oxide_device::{WriteBuffer, kernel, thread_idx};

/// # Safety
/// `output` holds one value per thread.
#[kernel]
pub unsafe fn reference_use(output: WriteBuffer<u32>, n: u32) {
    let mut value = n;
    let reference = &mut value;
    *reference += 1;
    unsafe { output.store_unchecked(thread_idx().x, value) };
}
