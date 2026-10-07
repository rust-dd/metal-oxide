#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[derive(Clone, Copy)]
pub struct SignedRecord {
    pub small: i8,
    pub wide: i16,
    pub pair: (i8, u16),
}

fn signed_update(mut value: SignedRecord) -> SignedRecord {
    let original = value;
    value.small = if original.wide < 0 { original.small.wrapping_add(1) } else { -original.small };
    value.wide = match original.wide {
        -32768 => 127,
        -1 => 255,
        0 => 0,
        _ => original.small as u8 as i16,
    };
    value.pair.0 = value.wide as i8;
    value
}

#[kernel]
pub unsafe fn signed_record(out8: WriteBuffer<i8>, out16: WriteBuffer<i16>, input: SignedRecord) {
    let value = signed_update(input);
    unsafe {
        out8.store_unchecked(0, value.small);
        out8.store_unchecked(1, value.pair.0);
        out16.store_unchecked(0, value.wide);
        out16.store_unchecked(1, value.pair.1 as i16);
    }
}

#[kernel]
pub unsafe fn signed_narrow(out8: WriteBuffer<i8>, out16: WriteBuffer<i16>, a: i8, b: i16, shift: u32) {
    unsafe {
        out8.store_unchecked(0, a.wrapping_add(127));
        out8.store_unchecked(1, a.wrapping_sub(127));
        out8.store_unchecked(2, a.wrapping_mul(-127));
        out8.store_unchecked(3, a.wrapping_shl(shift));
        out8.store_unchecked(4, a.wrapping_shr(shift));
        out8.store_unchecked(5, a.wrapping_neg());
        out8.store_unchecked(6, !a);
        out8.store_unchecked(7, b as i8);
        out16.store_unchecked(0, b.wrapping_add(32767));
        out16.store_unchecked(1, b.wrapping_sub(32767));
        out16.store_unchecked(2, b.wrapping_mul(-32767));
        out16.store_unchecked(3, b.wrapping_shl(shift));
        out16.store_unchecked(4, b.wrapping_shr(shift));
        out16.store_unchecked(5, b.wrapping_neg());
        out16.store_unchecked(6, !b);
        out16.store_unchecked(7, a as i16);
    }
}

#[kernel]
pub unsafe fn signed_flags(out8: WriteBuffer<i8>, out16: WriteBuffer<i16>, flags: WriteBuffer<u32>, a: i8, b: i16) {
    let (x, ox) = a.overflowing_mul(-127);
    let (y, oy) = b.overflowing_mul(-32767);
    unsafe {
        out8.store_unchecked(0, x);
        out16.store_unchecked(0, y);
        flags.store_unchecked(0, ox as u32);
        flags.store_unchecked(1, oy as u32);
    }
}
