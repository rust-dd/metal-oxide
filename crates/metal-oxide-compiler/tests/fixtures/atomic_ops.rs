#![no_std]
use core::sync::atomic::Ordering::Relaxed;
use metal_oxide_device::{AtomicBuffer, WriteBuffer, block_idx, kernel, thread_idx, threadgroup};

macro_rules! operations {
    ($name:ident, $ty:ty, $minimum:expr, ($($parameters:tt)*), $storage:expr) => {
        #[kernel]
        pub unsafe fn $name($($parameters)* out: WriteBuffer<$ty>, start: $ty) {
            let cell = $storage;
            unsafe {
                cell.store(0, start, Relaxed);
                out.store_unchecked(0, cell.load(0, Relaxed));
                out.store_unchecked(1, cell.exchange(0, 5, Relaxed));
                out.store_unchecked(2, cell.fetch_add(0, 3, Relaxed));
                out.store_unchecked(3, cell.fetch_sub(0, 1, Relaxed));
                out.store_unchecked(4, cell.fetch_min(0, $minimum, Relaxed));
                out.store_unchecked(5, cell.fetch_max(0, 9, Relaxed));
                out.store_unchecked(6, cell.fetch_and(0, 7, Relaxed));
                out.store_unchecked(7, cell.fetch_or(0, 4, Relaxed));
                out.store_unchecked(8, cell.fetch_xor(0, 3, Relaxed));
                let (observed, success) = cell.compare_exchange_weak(0, 99, 15, Relaxed, Relaxed);
                out.store_unchecked(9, observed);
                out.store_unchecked(10, success as $ty);
                let (observed, success) = cell.compare_exchange_weak(0, 6, 15, Relaxed, Relaxed);
                out.store_unchecked(11, observed);
                out.store_unchecked(12, success as $ty);
                out.store_unchecked(13, cell.load(0, Relaxed));
                cell.store(0, start, Relaxed);
                out.store_unchecked(14, cell.fetch_add(0, 1, Relaxed));
                out.store_unchecked(15, cell.load(0, Relaxed));
            }
        }
    };
}
operations!(atomic_u32, u32, 2, (cell: AtomicBuffer<u32>,), cell);
operations!(atomic_i32, i32, -2, (cell: AtomicBuffer<i32>,), cell);
operations!(shared_u32, u32, 2, (), threadgroup::shared_atomic::<u32, 1>());
operations!(shared_i32, i32, -2, (), threadgroup::shared_atomic::<i32, 1>());

#[kernel(block = (256, 1, 1))]
pub unsafe fn shared_counter(out: WriteBuffer<u32>) {
    let cells = threadgroup::shared_atomic::<u32, 2>();
    let lane = thread_idx().x;
    unsafe {
        if lane == 0 {
            cells.store(0, 0, Relaxed);
            cells.store(1, 0, Relaxed);
        }
        threadgroup::barrier();
        cells.fetch_add(lane & 1, 1, Relaxed);
        threadgroup::barrier();
        if lane == 0 {
            let offset = block_idx().x * 2;
            out.store_unchecked(offset, cells.load(0, Relaxed));
            out.store_unchecked(offset + 1, cells.load(1, Relaxed));
        }
    }
}
