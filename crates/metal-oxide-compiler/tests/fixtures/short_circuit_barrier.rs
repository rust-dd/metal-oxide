#![no_std]
use metal_oxide_device::{kernel, threadgroup};

#[kernel]
pub unsafe fn uniform_early_return(a: u32, b: u32) {
    if a == 0 && b == 0 {
        return;
    }
    unsafe { threadgroup::barrier() };
}
