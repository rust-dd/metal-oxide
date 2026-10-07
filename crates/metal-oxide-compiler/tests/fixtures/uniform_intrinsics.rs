#![no_std]

use metal_oxide_device::{
    F16, WriteBuffer, block_dim, block_idx, grid_dim, kernel, math, simdgroup, thread_idx,
    threadgroup,
};

fn root(value: f32) -> f32 {
    math::sqrt(value)
}

#[kernel(block = (32, 1, 1))]
pub unsafe fn uniform_intrinsics(output: WriteBuffer<u32>, value: f32) {
    let shared = threadgroup::shared::<u32, 32>();
    let lane = thread_idx().x;
    // SAFETY: each lane initializes its cell; every condition is uniform within a block.
    unsafe {
        shared.store_unchecked(lane, lane);
        if block_idx().x == 0 {
            threadgroup::barrier();
        }
        if block_dim().x == 32 {
            threadgroup::barrier();
        }
        if grid_dim().x == 3 {
            threadgroup::barrier();
        }
        if simdgroup::size() > 0 {
            threadgroup::barrier();
        }
        if simdgroup::groups_per_block() > 0 {
            threadgroup::barrier();
        }
        if root(value) == 2.0 {
            threadgroup::barrier();
        }
        if math::fma(value, 1.0, 0.0) == 4.0 {
            threadgroup::barrier();
        }
        if F16::from_f32(value).to_f32() == value {
            threadgroup::barrier();
        }
        threadgroup::barrier();
        output.store_unchecked(
            block_idx().x * 32 + lane,
            shared.load_unchecked((lane + 1) & 31),
        );
    }
}
