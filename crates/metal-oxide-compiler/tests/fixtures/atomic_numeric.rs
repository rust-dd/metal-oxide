#![no_std]
use core::sync::atomic::Ordering::Relaxed;
use metal_oxide_device::{AtomicBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn float_ops(cell: AtomicBuffer<f32>, out: WriteBuffer<f32>, start: f32) {
    unsafe {
        cell.store(0, start, Relaxed);
        out.store_unchecked(0, cell.load(0, Relaxed));
        out.store_unchecked(1, cell.exchange(0, 5.0, Relaxed));
        out.store_unchecked(2, cell.fetch_add(0, 3.0, Relaxed));
        out.store_unchecked(3, cell.fetch_sub(0, 1.0, Relaxed));
        let (old, success) = cell.compare_exchange_weak(0, 99.0, 15.0, Relaxed, Relaxed);
        out.store_unchecked(4, old);
        out.store_unchecked(5, success as u32 as f32);
        let (old, success) = cell.compare_exchange_weak(0, 7.0, 15.0, Relaxed, Relaxed);
        out.store_unchecked(6, old);
        out.store_unchecked(7, success as u32 as f32);
        out.store_unchecked(8, cell.load(0, Relaxed));
    }
}

#[kernel]
pub unsafe fn float_cas(cell: AtomicBuffer<f32>, out: WriteBuffer<u32>, expected: f32, desired: f32) {
    unsafe {
        let (old, success) = cell.compare_exchange_weak(0, expected, desired, Relaxed, Relaxed);
        out.store_unchecked(0, old.to_bits());
        out.store_unchecked(1, success as u32);
        out.store_unchecked(2, cell.load(0, Relaxed).to_bits());
    }
}

#[kernel]
pub unsafe fn float_contention(cell: AtomicBuffer<f32>) {
    unsafe { cell.fetch_add(0, 1.0, Relaxed); cell.fetch_sub(1, 1.0, Relaxed); }
}

#[kernel]
pub unsafe fn wide_contention(cell: AtomicBuffer<u64>) {
    let index = block_idx().x * block_dim().x + thread_idx().x;
    let value = (1_u64 << 48) + index as u64;
    unsafe { cell.min(0, value, Relaxed); cell.max(1, value, Relaxed); }
}
