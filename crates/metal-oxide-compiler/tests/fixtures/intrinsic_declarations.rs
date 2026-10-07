#![no_std]

use metal_oxide_device::{
    AtomicBuffer, F16, ReadBuffer, WriteBuffer, block_dim, block_idx, grid_dim, kernel, math,
    simdgroup, thread_idx, threadgroup,
};

#[kernel]
pub unsafe fn declarations(input: ReadBuffer<f32>, out: WriteBuffer<f32>, n: u32) {
    let i = block_idx()
        .x
        .wrapping_mul(block_dim().x)
        .wrapping_add(thread_idx().x);
    let groups = grid_dim().x;
    if i < n {
        unsafe {
            let x = input.load_unchecked(i);
            let half = F16::from_f32(math::sqrt(x));
            out.store_unchecked(i, math::fma(half.to_f32(), 2.0, groups as f32));
        }
    }
}

#[kernel(block = (32, 1, 1))]
pub unsafe fn cooperation(out: WriteBuffer<f32>, counts: AtomicBuffer<u32>) {
    let scratch = threadgroup::shared::<f32, 32>();
    let thread = thread_idx().x;
    unsafe {
        scratch.store_unchecked(thread, thread as f32);
        threadgroup::barrier();
        let sum = simdgroup::sum(scratch.load_unchecked(thread));
        let value = simdgroup::shuffle(sum, 0);
        let lane = simdgroup::lane_id();
        let width = simdgroup::size();
        let group = simdgroup::group_id();
        let groups = simdgroup::groups_per_block();
        let old = counts.fetch_add_relaxed(thread, 1);
        out.store_unchecked(thread, value + (lane + width + group + groups + old) as f32);
    }
}
