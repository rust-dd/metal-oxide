#![no_std]
use metal_oxide_device::{kernel, thread_index, threadgroup};
#[kernel]
pub unsafe fn divergent() {
    if thread_index().get() == 0 {
        // SAFETY: deliberately invalid participation for a compiler rejection test.
        unsafe { threadgroup::barrier(); }
    }
}
