#![no_std]

use metal_oxide_device::{
    ReadBuffer, WriteBuffer, block_idx, kernel, thread_idx, thread_index, threadgroup,
};

/// Sums each block's 256 inputs into the block's output element; inputs past the end count as
/// zero, and blocks past the end of `output` write nothing.
///
/// # Safety
/// Nothing writes `input` while the kernel runs.
#[kernel(block = (256, 1, 1))]
pub unsafe fn reduce(input: ReadBuffer<f32>, output: WriteBuffer<f32>) {
    let shared = threadgroup::shared::<f32, 256>();
    let lane = thread_idx().x;
    let block = block_idx().x;
    let input = input.as_slice();
    let i = thread_index().get();
    let value = if i < input.len() { input[i] } else { 0.0 };
    // SAFETY: lanes index the block's 256-element tile, and each block writes its own output.
    unsafe {
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
        if lane == 0 && (block as usize) < output.len() {
            output.store_unchecked(block, shared.load_unchecked(0));
        }
    }
}
