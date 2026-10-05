#![no_std]

use core::ops::Add;
use metal_oxide_device::{WriteBuffer, grid_dim, kernel};

fn sum<T: Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

fn bias<const N: u32>(value: u32) -> u32 {
    value + N
}

#[kernel]
pub unsafe fn helpers(float: WriteBuffer<f32>, integer: WriteBuffer<u32>) {
    // SAFETY: one thread writes float[grid_dim.x] and integer[0] within caller-provided buffers.
    unsafe {
        float.store_unchecked(grid_dim().x, sum(1.0, 2.0));
        integer.store_unchecked(0, bias::<3>(sum(1, 2)));
    }
}
