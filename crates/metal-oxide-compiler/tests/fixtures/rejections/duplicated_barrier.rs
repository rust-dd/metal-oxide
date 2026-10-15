#![no_std]
use metal_oxide_device::{kernel, threadgroup};

#[kernel]
pub unsafe fn invalid(a: u32, b: u32) {
    if a == 0 && b == 0 {
        return;
    }
    // Both short-circuit tests branch to this barrier, so codegen would emit it twice.
    unsafe { threadgroup::barrier() };
}
