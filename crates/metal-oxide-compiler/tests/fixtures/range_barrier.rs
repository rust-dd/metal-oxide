#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_idx, kernel, thread_idx, threadgroup};

/// # Safety
/// `input` and `output` hold one value per thread; blocks have 64 threads.
#[kernel(block = (64, 1, 1))]
pub unsafe fn shared_rotations(input: ReadBuffer<u32>, output: WriteBuffer<u32>, rounds: u32) {
    let shared = threadgroup::shared::<u32, 64>();
    let lane = thread_idx().x;
    let i = block_idx().x * 64 + lane;
    unsafe {
        let mut value = input.load_unchecked(i);
        for _ in 0..rounds {
            shared.store_unchecked(lane, value);
            threadgroup::barrier();
            value = shared.load_unchecked((lane + 1) % 64) + 1;
            threadgroup::barrier();
        }
        output.store_unchecked(i, value);
    }
}
