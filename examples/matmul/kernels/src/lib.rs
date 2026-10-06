#![no_std]

use metal_oxide_device::{
    ReadBuffer, WriteBuffer, block_idx, kernel, simdgroup, thread_idx, threadgroup,
};

/// # Safety
/// A and B cover n elements; output covers one element per SIMD group in the grid.
/// All buffers are distinct, and the block is 256x1x1.
#[kernel(block = (256, 1, 1))]
pub unsafe fn dot(a: ReadBuffer<f32>, b: ReadBuffer<f32>, output: WriteBuffer<f32>, n: u32) {
    let i = block_idx().x * 256 + thread_idx().x;
    let mut value = 0.0;
    // SAFETY: bounds are guarded and every lane participates in the reduction.
    unsafe {
        if i < n {
            value = a.load_unchecked(i) * b.load_unchecked(i);
        }
        let sum = simdgroup::sum(value);
        if simdgroup::lane_id() == 0 {
            output.store_unchecked(
                block_idx().x * simdgroup::groups_per_block() + simdgroup::group_id(),
                sum,
            );
        }
    }
}

/// # Safety
/// A, B, and output are distinct and cover rows*inner, inner*columns, and rows*columns elements.
/// Index products fit u32; the block is 16x16x1.
#[kernel(block = (16, 16, 1))]
pub unsafe fn matmul(
    a: ReadBuffer<f32>,
    b: ReadBuffer<f32>,
    output: WriteBuffer<f32>,
    rows: u32,
    columns: u32,
    inner: u32,
) {
    let tile_a = threadgroup::shared::<f32, 256>();
    let tile_b = threadgroup::shared::<f32, 256>();
    let thread = thread_idx();
    let block = block_idx();
    let row = block.y * 16 + thread.y;
    let column = block.x * 16 + thread.x;
    let slot = thread.y * 16 + thread.x;
    let mut result = 0.0;
    let mut base = 0;
    // SAFETY: all shared cells are initialized and both barriers are reached by the whole block.
    unsafe {
        while base < inner {
            let mut x = 0.0;
            let mut y = 0.0;
            if row < rows && base + thread.x < inner {
                x = a.load_unchecked(row * inner + base + thread.x);
            }
            if column < columns && base + thread.y < inner {
                y = b.load_unchecked((base + thread.y) * columns + column);
            }
            tile_a.store_unchecked(slot, x);
            tile_b.store_unchecked(slot, y);
            threadgroup::barrier();
            let mut k = 0;
            while k < 16 {
                result += tile_a.load_unchecked(thread.y * 16 + k)
                    * tile_b.load_unchecked(k * 16 + thread.x);
                k += 1;
            }
            threadgroup::barrier();
            base += 16;
        }
        if row < rows && column < columns {
            output.store_unchecked(row * columns + column, result);
        }
    }
}
