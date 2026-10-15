#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn bit_ranges(out: WriteBuffer<i32>, a: u32, b: i32, c: i32) {
    unsafe {
        out.store_unchecked(0, (a / (a.rotate_left(3) | 1)) as i32);
        out.store_unchecked(1, b / ((c & 0x3fff_ffff) | 1));
        out.store_unchecked(2, b % (c | 1));
    }
}
