#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

unsafe fn scale<const BLOCK: u32, const SCALE: u32>(
    input: ReadBuffer<f32>,
    output: WriteBuffer<f32>,
    n: u32,
) {
    let i = block_idx().x * BLOCK + thread_idx().x;
    if i < n {
        // SAFETY: entrypoint callers supply n elements and one writer per index.
        unsafe {
            output.store_unchecked(
                i,
                pipeline_device_math::first::<SCALE>(input.load_unchecked(i)),
            )
        };
    }
}

/// # Safety
/// Buffers cover n elements, do not overlap, and the launch uses 128 threads per block.
#[kernel(block = (128, 1, 1))]
pub unsafe fn scale_128(input: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
    // SAFETY: the entrypoint contract fixes the index calculation and buffer bounds.
    unsafe { scale::<128, 2>(input, output, n) };
}

/// # Safety
/// Buffers cover n elements, do not overlap, and the launch uses 256 threads per block.
#[kernel(block = (256, 1, 1))]
pub unsafe fn scale_256(input: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
    // SAFETY: the entrypoint contract fixes the index calculation and buffer bounds.
    unsafe { scale::<256, 4>(input, output, n) };
}

/// # Safety
/// Buffers cover n elements, do not overlap, and the launch is one dimensional.
#[kernel]
pub unsafe fn finish(input: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: the contract gives each active thread one initialized input and distinct output.
        unsafe { output.store_unchecked(i, pipeline_device_math::second(input.load_unchecked(i))) };
    }
}
