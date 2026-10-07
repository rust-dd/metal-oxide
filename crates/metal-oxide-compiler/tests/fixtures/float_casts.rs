#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn float_casts(input: ReadBuffer<f32>, out_i32: WriteBuffer<i32>, out_u32: WriteBuffer<u32>, out_i8: WriteBuffer<i8>, out_u8: WriteBuffer<u8>, out_i16: WriteBuffer<i16>, out_u16: WriteBuffer<u16>, high: WriteBuffer<u32>, n: u32) {
    let i = block_idx().x.wrapping_mul(block_dim().x).wrapping_add(thread_idx().x);
    if i < n {
        unsafe {
            let value = input.load_unchecked(i);
            out_i32.store_unchecked(i, value as i32);
            out_u32.store_unchecked(i, value as u32);
            out_i8.store_unchecked(i, value as i8);
            out_u8.store_unchecked(i, value as u8);
            out_i16.store_unchecked(i, value as i16);
            out_u16.store_unchecked(i, value as u16);
            high.store_unchecked(i, ((value as usize) >> 32) as u32);
        }
    }
}
