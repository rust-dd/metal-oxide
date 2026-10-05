#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_idx, kernel, thread_idx, threadgroup};

#[kernel(block = (256, 1, 1))]
pub unsafe fn reduce(input: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
    let shared = threadgroup::shared::<f32, 256>();
    let lane = thread_idx().x;
    let i = block_idx().x * 256 + lane;
    let mut value = 0.0;
    // SAFETY: the host supplies n initialized inputs, one output per block, and a 256-thread block.
    unsafe {
        if i < n {
            value = input.load_unchecked(i);
        }
        shared.store_unchecked(lane, value);
        threadgroup::barrier();
        let mut stride = 128;
        while stride > 0 {
            if lane < stride {
                let sum = shared.load_unchecked(lane) + shared.load_unchecked(lane + stride);
                shared.store_unchecked(lane, sum);
            }
            threadgroup::barrier();
            stride >>= 1;
        }
        if lane == 0 {
            output.store_unchecked(block_idx().x, shared.load_unchecked(0));
        }
    }
}
