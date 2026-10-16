#![no_std]
use core::sync::atomic::Ordering::{Acquire, Release, AcqRel, Relaxed};
use metal_oxide_device::{AtomicBuffer, WriteBuffer, kernel};

#[kernel]
pub unsafe fn ordered(cell: AtomicBuffer<u32>, out: WriteBuffer<u32>) {
    unsafe {
        cell.store(0, 2, Release);
        out.store_unchecked(0, cell.load(0, Acquire));
        out.store_unchecked(1, cell.fetch_add(0, 1, AcqRel));
        let (old, _) = cell.compare_exchange_weak(0, 3, 4, AcqRel, Acquire);
        out.store_unchecked(2, old);
        let (old, _) = cell.compare_exchange_weak(0, 4, 5, Release, Relaxed);
        out.store_unchecked(3, old);
    }
}
