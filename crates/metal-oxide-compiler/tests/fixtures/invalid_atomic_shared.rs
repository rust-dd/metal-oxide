#![no_std]
use core::sync::atomic::Ordering::Relaxed;
use metal_oxide_device::{kernel, threadgroup};
#[kernel]
pub unsafe fn invalid_atomic() {
    let cells = threadgroup::shared_atomic::<f32, 1>();
    unsafe { cells.store(0, 0.0, Relaxed); }
}
