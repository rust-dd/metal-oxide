#![no_std]
use metal_oxide_device::{kernel, threadgroup};

#[kernel]
pub unsafe fn huge_threadgroup(value: u32) {
    let scratch = threadgroup::shared::<u32, { 1 << 32 }>();
    unsafe { scratch.store_unchecked(0, value) };
}
