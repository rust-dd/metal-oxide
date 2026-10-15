#![no_std]
use metal_oxide_device::{kernel, threadgroup};
#[kernel]
pub unsafe fn shared_in_loop(count: u32) {
    let mut index = 0;
    while index < count {
        let scratch = threadgroup::shared::<u32, 4>();
        unsafe { scratch.store_unchecked(0, index); }
        index += 1;
    }
}
