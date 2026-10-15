#![no_std]
use metal_oxide_device::{kernel, threadgroup};

#[kernel]
pub unsafe fn invalid(value: u32) {
    let references = threadgroup::shared::<&u32, 4>();
    unsafe { references.store_unchecked(0, &value) };
}
