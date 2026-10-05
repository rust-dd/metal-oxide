#![no_std]
use metal_oxide_device::{kernel, thread_idx, threadgroup};
#[kernel]
pub unsafe fn divergent() {
    if thread_idx().x == 0 {
        // SAFETY: deliberately invalid participation for a compiler rejection test.
        unsafe { threadgroup::barrier(); }
    }
}
