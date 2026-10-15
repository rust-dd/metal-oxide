#![no_std]
use metal_oxide_device::{kernel, threadgroup};

#[kernel]
pub unsafe fn invalid() {
    let _tile = threadgroup::shared::<u32, 0>();
}
