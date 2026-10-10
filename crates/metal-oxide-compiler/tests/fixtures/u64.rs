#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

fn sm64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[kernel]
pub unsafe fn splitmix(out: WriteBuffer<u64>, base: u64, seed: u32, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        let cell = base.wrapping_add(i as u64);
        let key = cell ^ (seed as u64).wrapping_mul(0x9e3779b97f4a7c15);
        unsafe { out.store_unchecked(i, sm64(key)) };
    }
}

#[kernel]
pub unsafe fn wide_casts(input: ReadBuffer<f32>, output: WriteBuffer<u64>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        unsafe { output.store_unchecked(i, input.load_unchecked(i) as u64) };
    }
}

#[kernel]
pub unsafe fn wide_ops(out: WriteBuffer<u64>, a: u64, b: u64, shift: u32, signed: i32) {
    let (sum, add_overflow) = a.overflowing_add(b);
    let (product, mul_overflow) = a.overflowing_mul(b);
    let (difference, sub_overflow) = a.overflowing_sub(b);
    unsafe {
        out.store_unchecked(0, sum);
        out.store_unchecked(1, product);
        out.store_unchecked(2, difference);
        out.store_unchecked(3, (!a).wrapping_shl(shift));
        out.store_unchecked(4, a.wrapping_shr(shift));
        out.store_unchecked(5, signed as u64);
        out.store_unchecked(6, a as u32 as u64);
        out.store_unchecked(7, add_overflow as u64);
        out.store_unchecked(8, mul_overflow as u64);
        out.store_unchecked(9, sub_overflow as u64);
        out.store_unchecked(10, if b == 0 { 0 } else { a / b });
        out.store_unchecked(11, if b == 0 { 0 } else { a % b });
        out.store_unchecked(12, a as f32 as u64);
        out.store_unchecked(13, a as i32 as u64);
        out.store_unchecked(
            14,
            match a {
                u64::MAX => 7,
                0x100000000 => 8,
                _ => 9,
            },
        );
    }
}
