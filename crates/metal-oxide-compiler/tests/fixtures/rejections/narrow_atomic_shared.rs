#![no_std]
use metal_oxide_device::{kernel, threadgroup};
#[kernel]
pub unsafe fn narrow_atomic() {
    let cells = threadgroup::shared_atomic::<u8, 1>();
    unsafe { cells.store(0, 0, core::sync::atomic::Ordering::Relaxed); }
}
