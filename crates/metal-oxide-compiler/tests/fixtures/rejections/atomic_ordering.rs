#![no_std]

use core::sync::atomic::Ordering::{Acquire, Relaxed};
use metal_oxide_device::{AtomicBuffer, kernel};

#[kernel]
pub unsafe fn rejected(cell: AtomicBuffer<u32>, flag: u32) {
    let order = if flag == 0 { Relaxed } else { Acquire };
    // SAFETY: the caller supplies one atomic element and launches one thread.
    unsafe { cell.load(0, order) };
}
