#![no_std]
use metal_oxide_device::{WriteBuffer, kernel, simdgroup, thread_idx};
#[kernel]
pub unsafe fn divergent(output: WriteBuffer<f32>) {
    if thread_idx().x == 0 {
        // SAFETY: deliberately invalid participation for a compiler rejection test.
        unsafe { output.store_unchecked(0, simdgroup::sum(1.0)); }
    }
}
