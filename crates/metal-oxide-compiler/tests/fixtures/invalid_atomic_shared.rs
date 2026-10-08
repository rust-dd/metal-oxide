#![no_std]
use metal_oxide_device::{kernel, threadgroup};
#[kernel]
pub unsafe fn invalid_atomic() {
    let cells = threadgroup::shared_atomic::<f32, 1>();
    unsafe { cells.store_relaxed(0, 0.0); }
}
