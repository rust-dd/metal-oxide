#![no_std]
use metal_oxide_device::{
    ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx, threadgroup,
};

fn index() -> u32 {
    block_idx().x * block_dim().x + thread_idx().x
}

#[kernel]
pub unsafe fn lookup(indices: ReadBuffer<u32>, out: WriteBuffer<u32>, n: u32) {
    let i = index();
    if i < n {
        let values = [10_u32, 20, 30, 40];
        unsafe { out.store_unchecked(i, values[indices.load_unchecked(i) as usize]) };
    }
}

#[kernel]
pub unsafe fn fill(out: WriteBuffer<u32>, value: u32, n: u32) {
    let i = index();
    if i < n {
        unsafe { out.store_unchecked(i, value) };
    }
}

#[kernel]
pub unsafe fn arithmetic(values: ReadBuffer<u32>, out: WriteBuffer<u32>, divisor: u32, n: u32) {
    let i = index();
    if i < n {
        let value = unsafe { values.load_unchecked(i) };
        assert!(value != 7, "seven is not allowed");
        unsafe { out.store_unchecked(i, (value + 1) / divisor) };
    }
}

#[kernel(block = (64, 1, 1))]
pub unsafe fn reverse_scaled(out: WriteBuffer<u32>, divisor: u32) {
    let shared = threadgroup::shared::<u32, 64>();
    let lane = thread_idx().x;
    // Every thread fails this uniform check together, so none waits at the barrier.
    let scale = 64 / divisor;
    // SAFETY: each lane writes its own element, and the whole block reaches the barrier.
    unsafe {
        shared.store_unchecked(lane, lane.wrapping_mul(scale));
        threadgroup::barrier();
        out.store_unchecked(block_idx().x * 64 + lane, shared.load_unchecked(63 - lane));
    }
}
