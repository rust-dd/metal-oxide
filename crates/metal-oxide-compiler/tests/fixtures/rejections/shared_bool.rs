#![no_std]
use metal_oxide_device::{kernel, threadgroup};

#[kernel]
pub unsafe fn invalid(value: u32) {
    let flags = threadgroup::shared::<bool, 4>();
    unsafe { flags.store_unchecked(0, value == 0) };
}
