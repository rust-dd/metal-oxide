#![no_std]
use metal_oxide_device::{kernel, thread_idx, threadgroup};
#[kernel]
pub unsafe fn varying_loop() {
    let mut i = 0;
    while i < thread_idx().x {
        // SAFETY: deliberately invalid participation for a compiler rejection test.
        unsafe { threadgroup::barrier(); }
        i += 1;
    }
}
