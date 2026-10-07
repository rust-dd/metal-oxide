#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

fn quotient(value: i32, divisor: i32) -> i32 {
    value / divisor + 1
}

#[kernel]
pub unsafe fn guarded_division(out: WriteBuffer<i32>, value: i32, divisor: i32) {
    if divisor == 0 { return; }
    if value == i32::MIN && divisor == -1 { return; }
    unsafe {
        out.store_unchecked(0, value / divisor);
        out.store_unchecked(1, value % divisor);
        out.store_unchecked(2, quotient(value, 4));
    }
}

#[kernel]
pub unsafe fn checked_range(out: WriteBuffer<u32>, value: u32) {
    if value < 16 {
        let mut result = value + 1;
        if value < 8 { result = result * 2; } else { result = result + 8; }
        unsafe { out.store_unchecked(0, result); }
    }
}

#[kernel]
pub unsafe fn checked_array_loop(out: WriteBuffer<u32>) {
    let mut values = [10_u32, 20, 30, 40];
    let mut i = 0_u32;
    while i < 4 {
        values[i as usize] = values[i as usize] + 1;
        unsafe { out.store_unchecked(i, values[i as usize]); }
        i = i + 1;
    }
}

#[kernel]
pub unsafe fn guarded_index(out: WriteBuffer<u32>, index: u32) {
    let mut values = [10_u32, 20, 30, 40];
    if index < 4 {
        values[index as usize] = values[index as usize] + 1;
        unsafe { out.store_unchecked(0, values[index as usize]); }
    }
}

#[kernel]
pub unsafe fn narrow_division(out8: WriteBuffer<i8>, out16: WriteBuffer<i16>, a: i8, b: i16) {
    unsafe {
        out8.store_unchecked(0, a / 3);
        out8.store_unchecked(1, a % 3);
        out16.store_unchecked(0, b / 3);
        out16.store_unchecked(1, b % 3);
    }
}
