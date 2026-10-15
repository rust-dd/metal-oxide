#![no_std]
use metal_oxide_device::{kernel, threadgroup};
fn stage(value: u32) {
    let scratch = threadgroup::shared::<u32, 4>();
    unsafe { scratch.store_unchecked(0, value); }
}
#[kernel]
pub unsafe fn shared_in_helper(value: u32) {
    stage(value);
}
