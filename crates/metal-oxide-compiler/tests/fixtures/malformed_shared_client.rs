#![no_std]

use metal_oxide_device::{kernel, threadgroup};

#[kernel(block = (32, 1, 1))]
pub unsafe fn invalid_allocation() {
    #[cfg(reversed)]
    let _ = threadgroup::shared::<32, f32>();
    #[cfg(any(fixed_length, fixed_element))]
    let _ = threadgroup::shared::<f32, 32>();
}
