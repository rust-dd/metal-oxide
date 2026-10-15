#![no_std]
use metal_oxide_device::{kernel, threadgroup};

#[kernel]
pub unsafe fn invalid(value: u32) {
    let pairs = threadgroup::shared::<(u32, u32), 4>();
    unsafe { pairs.store_unchecked(0, (value, value)) };
}
