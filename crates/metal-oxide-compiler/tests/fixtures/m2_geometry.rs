#![no_std]

use metal_oxide_device::{Dim3, WriteBuffer, block_dim, block_idx, grid_dim, kernel, thread_idx};

fn grid() -> Dim3 {
    grid_dim()
}

#[kernel]
pub unsafe fn geometry(out: WriteBuffer<u32>) {
    let thread = thread_idx();
    let block = block_idx();
    let dim = block_dim();
    let grid = grid();
    let x = block.x * dim.x + thread.x;
    let y = block.y * dim.y + thread.y;
    let z = block.z * dim.z + thread.z;
    let width = grid.x * dim.x;
    let height = grid.y * dim.y;
    let offset = (x + width * (y + height * z)) * 15;
    // SAFETY: each global thread owns a distinct 15-element record in the output.
    unsafe {
        out.store_unchecked(offset, x);
        out.store_unchecked(offset + 1, y);
        out.store_unchecked(offset + 2, z);
        out.store_unchecked(offset + 3, thread.x);
        out.store_unchecked(offset + 4, thread.y);
        out.store_unchecked(offset + 5, thread.z);
        out.store_unchecked(offset + 6, block.x);
        out.store_unchecked(offset + 7, block.y);
        out.store_unchecked(offset + 8, block.z);
        out.store_unchecked(offset + 9, dim.x);
        out.store_unchecked(offset + 10, dim.y);
        out.store_unchecked(offset + 11, dim.z);
        out.store_unchecked(offset + 12, grid.x);
        out.store_unchecked(offset + 13, grid.y);
        out.store_unchecked(offset + 14, grid.z);
    }
}
