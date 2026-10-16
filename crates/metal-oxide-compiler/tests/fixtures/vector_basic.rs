#![no_std]
#![feature(portable_simd)]
use core::simd::{Simd, Mask, prelude::*};
use metal_oxide_device::{kernel, ReadBuffer, WriteBuffer};

#[kernel]
pub unsafe fn vector_f32(input: ReadBuffer<[f32; 4]>, out: WriteBuffer<[f32; 4]>) {
    let a = Simd::from_array(unsafe { input.load_unchecked(0) });
    let b = Simd::splat(2.0);
    let mask = a.simd_lt(b);
    let result = mask.select((a + b) * b, (a - b) / b).abs();
    unsafe { out.store_unchecked(0, result.to_array()); }
}

#[kernel]
pub unsafe fn scalar_f32(input: ReadBuffer<[f32; 4]>, out: WriteBuffer<[f32; 4]>) {
    let a = unsafe { input.load_unchecked(0) };
    let scalar = |value: f32| {
        if value < 2.0 { ((value + 2.0) * 2.0).abs() } else { ((value - 2.0) / 2.0).abs() }
    };
    let result = [scalar(a[0]), scalar(a[1]), scalar(a[2]), scalar(a[3])];
    unsafe { out.store_unchecked(0, result); }
}

#[kernel]
pub unsafe fn vector_i32(input: ReadBuffer<[i32; 3]>, out: WriteBuffer<[i32; 3]>) {
    let mut a = Simd::from_array(unsafe { input.load_unchecked(0) });
    a[1] = a[0];
    let b = Simd::splat(2);
    let result = a.saturating_add(b).simd_max(-a) / b;
    unsafe { out.store_unchecked(0, result.to_array()); }
}

#[kernel]
pub unsafe fn vector_mask(input: ReadBuffer<[u8; 2]>, out: WriteBuffer<u64>) {
    let a = Simd::from_array(unsafe { input.load_unchecked(0) });
    let mask = a.simd_eq(Simd::splat(0));
    let reconstructed = Mask::<i8, 2>::from_bitmask(mask.to_bitmask());
    let selected = reconstructed.select(a.reverse_bits(), a.count_ones());
    unsafe { out.store_unchecked(0, selected.reduce_sum() as u64); }
}

#[kernel]
pub unsafe fn vector_division(input: ReadBuffer<[i32; 4]>, out: WriteBuffer<[i32; 4]>, remainder: u32) {
    let a = Simd::from_array(unsafe { input.load_unchecked(0) });
    let b = Simd::from_array(unsafe { input.load_unchecked(1) });
    let result = if remainder != 0 { a % b } else { a / b };
    unsafe { out.store_unchecked(0, result.to_array()); }
}

#[kernel]
pub unsafe fn vector_index(input: ReadBuffer<[i32; 3]>, out: WriteBuffer<[i32; 3]>, index: u64) {
    let mut a = Simd::from_array(unsafe { input.load_unchecked(0) });
    let index = index as usize;
    a[index] = a[index].wrapping_add(1);
    unsafe { out.store_unchecked(0, a.to_array()); }
}
