#![no_std]

use metal_oxide_device::{
    DynamicThreadgroupBuffer, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx,
    threadgroup::{barrier, shared},
};

unsafe fn reduce(
    values: DynamicThreadgroupBuffer<f32>,
    tags: DynamicThreadgroupBuffer<(u8, u64)>,
) -> f32 {
    unsafe { barrier() };
    let mut sum = 0.0;
    if thread_idx().x == 0 {
        let mut i = 0;
        while i < block_dim().x {
            unsafe {
                let tag = tags.load_unchecked(i);
                sum = sum + values.load_unchecked(i) + tag.0 as f32 + tag.1 as f32;
            }
            i += 1;
        }
    }
    sum
}

#[kernel]
pub unsafe fn dynamic_reduce(
    input: ReadBuffer<f32>,
    scratch: DynamicThreadgroupBuffer<f32>,
    bias: f32,
    output: WriteBuffer<f32>,
    tags: DynamicThreadgroupBuffer<(u8, u64)>,
    n: u32,
) {
    let lane = thread_idx().x;
    let i = block_idx().x * block_dim().x + lane;
    let header = shared::<u32, 1>();
    unsafe {
        if lane == 0 {
            header.store_unchecked(0, 7);
        }
        if i < n {
            scratch.store_unchecked(lane, input.load_unchecked(i) + bias);
            tags.store_unchecked(lane, ((i % 17) as u8, i as u64));
        } else {
            scratch.store_unchecked(lane, 0.0);
            tags.store_unchecked(lane, (0, 0));
        }
        let sum = reduce(scratch, tags);
        if lane == 0 {
            output.store_unchecked(block_idx().x, sum + header.load_unchecked(0) as f32);
        }
    }
}
