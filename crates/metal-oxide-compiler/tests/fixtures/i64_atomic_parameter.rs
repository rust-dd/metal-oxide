#![no_std]
use metal_oxide_device::{AtomicBuffer, kernel};

#[kernel]
pub unsafe fn rejected(cell: AtomicBuffer<i64>) {
    unsafe { cell.store(0, 1, core::sync::atomic::Ordering::Relaxed) };
}
