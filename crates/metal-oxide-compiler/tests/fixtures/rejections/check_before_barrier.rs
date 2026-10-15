#![no_std]
use metal_oxide_device::{WriteBuffer, kernel, thread_idx, threadgroup};
#[kernel]
pub unsafe fn check_before_barrier(out: WriteBuffer<u32>) {
    let values = [1_u32, 2, 3, 4];
    // Threads past the array fail the bounds check and skip the barrier.
    let value = values[thread_idx().x as usize];
    unsafe { threadgroup::barrier(); }
    unsafe { out.store_unchecked(0, value); }
}
