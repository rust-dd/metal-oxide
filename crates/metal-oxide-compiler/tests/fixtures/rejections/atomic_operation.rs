#![no_std]

use core::sync::atomic::Ordering::Relaxed;
use metal_oxide_device::{AtomicBuffer, kernel};

#[kernel]
pub unsafe fn rejected(cell: AtomicBuffer<f32>) {
    // SAFETY: the caller supplies one atomic element and launches one thread.
    unsafe { cell.fetch_min(0, 1.0, Relaxed) };
}
