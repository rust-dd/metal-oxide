#![no_std]

use metal_oxide_device::{AtomicBuffer, ReadBuffer, WriteBuffer, block_idx, kernel, thread_idx, threadgroup};

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

#[kernel(block = (16, 16, 1))]
pub unsafe fn transpose(input: ReadBuffer<f32>, output: WriteBuffer<f32>, width: u32, height: u32) {
    let tile = threadgroup::shared::<f32, 272>();
    let thread = thread_idx();
    let block = block_idx();
    let x = block.x * 16 + thread.x;
    let y = block.y * 16 + thread.y;
    // SAFETY: buffers cover width*height; each tile cell is initialized before the uniform barrier.
    unsafe {
        let mut value = 0.0;
        if x < width && y < height {
            value = input.load_unchecked(y * width + x);
        }
        tile.store_unchecked(thread.y * 17 + thread.x, value);
        threadgroup::barrier();
        let out_x = block.y * 16 + thread.x;
        let out_y = block.x * 16 + thread.y;
        if out_x < height && out_y < width {
            output.store_unchecked(out_y * height + out_x, tile.load_unchecked(thread.x * 17 + thread.y));
        }
    }
}

#[kernel]
pub unsafe fn counter(output: AtomicBuffer<u32>) {
    // SAFETY: output[0] is initialized, and all concurrent accesses are atomic.
    unsafe { output.fetch_add_relaxed(0, 1); }
}
