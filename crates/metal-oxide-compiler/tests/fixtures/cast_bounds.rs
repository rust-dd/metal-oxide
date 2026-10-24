#![no_std]

use metal_oxide_device::{
    ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx, threadgroup,
};

#[kernel]
pub unsafe fn guarded_cast(input: ReadBuffer<u32>, output: WriteBuffer<u32>) {
    let i = thread_idx().x;
    let values = input.as_slice();
    if (i as usize) < values.len() {
        unsafe { output.store_unchecked(i, values[i as usize]) };
    }
}

#[kernel]
pub unsafe fn narrowed_cast(input: ReadBuffer<u32>, output: WriteBuffer<u32>, index: u64) {
    let values = input.as_slice();
    if (index as u8 as usize) < values.len() {
        unsafe { output.store_unchecked(0, values[index as u8 as usize]) };
    }
}

#[kernel]
pub unsafe fn loop_cast(input: ReadBuffer<u32>, output: WriteBuffer<u32>, count: u32) {
    let values = input.as_slice();
    let mut sum = 0_u32;
    let mut i = 0_u32;
    while i < count {
        if (i as usize) < values.len() {
            sum = sum.wrapping_add(values[i as usize]);
        }
        i += 1;
    }
    unsafe { output.store_unchecked(0, sum) };
}

#[kernel(block = (64, 1, 1))]
pub unsafe fn reduce_cast(input: ReadBuffer<u32>, output: WriteBuffer<u32>) {
    let lane = thread_idx().x;
    let i = block_idx().x * block_dim().x + lane;
    let values = input.as_slice();
    let value = if (i as usize) < values.len() {
        values[i as usize]
    } else {
        0
    };
    let scratch = threadgroup::shared::<u32, 64>();
    // SAFETY: each lane initializes its slot; all lanes reach the barrier before lane zero reads.
    unsafe {
        scratch.store_unchecked(lane, value);
        threadgroup::barrier();
        if lane == 0 {
            let mut sum = 0_u32;
            for j in 0..64 {
                sum = sum.wrapping_add(scratch.load_unchecked(j));
            }
            output.store_unchecked(block_idx().x, sum);
        }
    }
}
