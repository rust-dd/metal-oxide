#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn merged_division(out: WriteBuffer<i16>, input: u32) {
    let offset0 = if input & 1 == 0 { 2_i16 } else { 5_i16 };
    let mut value = 1_i16;
    for _ in 0..4_u32 {
        value = value.wrapping_add(value / input as i16);
        value = match value & 3 {
            0 => i16::MAX / (value | 1),
            1 => value.wrapping_add(value),
            2 => i16::MIN,
            _ => 1,
        }
        .wrapping_shl(3);
    }
    unsafe { out.store_unchecked(0, value.wrapping_add(offset0)) };
}
