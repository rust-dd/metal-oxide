#![no_std]

use metal_oxide_device::{ReadWriteBuffer, block_dim, block_idx, kernel, thread_idx};

unsafe fn butterfly(values: ReadWriteBuffer<f32>, index: u32) {
    unsafe {
        let a = values.load_unchecked(index);
        let b = values.load_unchecked(index + 1);
        values.store_unchecked(index, a + b);
        values.store_unchecked(index + 1, a - b);
    }
}

#[kernel]
pub unsafe fn fft_stage(values: ReadWriteBuffer<f32>, pairs: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < pairs {
        unsafe { butterfly(values, i * 2) };
    }
}
