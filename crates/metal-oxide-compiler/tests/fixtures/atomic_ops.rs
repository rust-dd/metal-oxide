#![no_std]
use metal_oxide_device::{AtomicBuffer, WriteBuffer, block_idx, kernel, thread_idx, threadgroup};

macro_rules! operations {
    ($name:ident, $ty:ty, $minimum:expr, ($($parameters:tt)*), $storage:expr) => {
        #[kernel]
        pub unsafe fn $name($($parameters)* out: WriteBuffer<$ty>, start: $ty) {
            let cell = $storage;
            unsafe {
                cell.store_relaxed(0, start);
                out.store_unchecked(0, cell.load_relaxed(0));
                out.store_unchecked(1, cell.exchange_relaxed(0, 5));
                out.store_unchecked(2, cell.fetch_add_relaxed(0, 3));
                out.store_unchecked(3, cell.fetch_sub_relaxed(0, 1));
                out.store_unchecked(4, cell.fetch_min_relaxed(0, $minimum));
                out.store_unchecked(5, cell.fetch_max_relaxed(0, 9));
                out.store_unchecked(6, cell.fetch_and_relaxed(0, 7));
                out.store_unchecked(7, cell.fetch_or_relaxed(0, 4));
                out.store_unchecked(8, cell.fetch_xor_relaxed(0, 3));
                let (observed, success) = cell.compare_exchange_weak_relaxed(0, 99, 15);
                out.store_unchecked(9, observed);
                out.store_unchecked(10, success as $ty);
                let (observed, success) = cell.compare_exchange_weak_relaxed(0, 6, 15);
                out.store_unchecked(11, observed);
                out.store_unchecked(12, success as $ty);
                out.store_unchecked(13, cell.load_relaxed(0));
                cell.store_relaxed(0, start);
                out.store_unchecked(14, cell.fetch_add_relaxed(0, 1));
                out.store_unchecked(15, cell.load_relaxed(0));
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
            cells.store_relaxed(0, 0);
            cells.store_relaxed(1, 0);
        }
        threadgroup::barrier();
        cells.fetch_add_relaxed(lane & 1, 1);
        threadgroup::barrier();
        if lane == 0 {
            let offset = block_idx().x * 2;
            out.store_unchecked(offset, cells.load_relaxed(0));
            out.store_unchecked(offset + 1, cells.load_relaxed(1));
        }
    }
}
