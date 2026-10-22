#![no_std]

use core::num::NonZeroU32;
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn powers(out: WriteBuffer<u32>, a: u32, b: i32, exponent: u32) {
    let values = [
        a.pow(exponent),
        a.checked_pow(exponent).unwrap_or(0),
        a.checked_pow(exponent).is_some() as u32,
        a.wrapping_pow(exponent),
        a.saturating_pow(exponent),
        b.pow(exponent) as u32,
        b.checked_pow(exponent).unwrap_or(0) as u32,
        b.checked_pow(exponent).is_some() as u32,
        b.wrapping_pow(exponent) as u32,
        b.saturating_pow(exponent) as u32,
        (a as u8).pow(exponent) as u32,
        (a as u8).checked_pow(exponent).unwrap_or(0) as u32,
        (a as u8).wrapping_pow(exponent) as u32,
        (a as u8).saturating_pow(exponent) as u32,
        (b as i16).pow(exponent) as u32,
        (b as i16).checked_pow(exponent).unwrap_or(0) as u32,
        (b as i16).wrapping_pow(exponent) as u32,
        (b as i16).saturating_pow(exponent) as u32,
    ];
    for i in 0..18 {
        unsafe { out.store_unchecked(i as u32, values[i]) };
    }
}

#[kernel]
pub unsafe fn wide_powers(out: WriteBuffer<u64>, a: u64, exponent: u32) {
    let values = [
        a.pow(exponent),
        a.checked_pow(exponent).unwrap_or(0),
        a.checked_pow(exponent).is_some() as u64,
        a.wrapping_pow(exponent),
        a.saturating_pow(exponent),
    ];
    for i in 0..5 {
        unsafe { out.store_unchecked(i as u32, values[i]) };
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    major: u32,
    minor: i32,
}

#[kernel]
pub unsafe fn comparisons(out: WriteBuffer<i32>, a: u32, b: u32, c: i32, d: i32) {
    let left = Key { major: a, minor: c };
    let right = Key { major: b, minor: d };
    let values = [
        a.cmp(&b) as i32,
        a.partial_cmp(&b).unwrap() as i32,
        c.cmp(&d) as i32,
        c.partial_cmp(&d).unwrap() as i32,
        left.cmp(&right) as i32,
        left.partial_cmp(&right).unwrap() as i32,
        c.signum(),
    ];
    for i in 0..7 {
        unsafe { out.store_unchecked(i as u32, values[i]) };
    }
}

#[kernel]
pub unsafe fn primitive_ordering(out: WriteBuffer<i32>, a: u64, b: u64) {
    let values = [
        (a as u8).cmp(&(b as u8)) as i32,
        (a as i8).cmp(&(b as i8)) as i32,
        (a as u16).cmp(&(b as u16)) as i32,
        (a as i16).cmp(&(b as i16)) as i32,
        (a as u32).cmp(&(b as u32)) as i32,
        (a as i32).cmp(&(b as i32)) as i32,
        a.cmp(&b) as i32,
        (a as usize).cmp(&(b as usize)) as i32,
        (a as isize).cmp(&(b as isize)) as i32,
        (a != 0).cmp(&(b != 0)) as i32,
    ];
    for i in 0..10 {
        unsafe { out.store_unchecked(i as u32, values[i]) };
    }
}

#[kernel]
pub unsafe fn float_methods(out: WriteBuffer<u32>, a: f32, b: f32) {
    let values = [
        a.copysign(b).to_bits(),
        a.signum().to_bits(),
        a.total_cmp(&b) as i32 as u32,
        a.partial_cmp(&b).map(|order| order as i32 as u32).unwrap_or(2),
    ];
    for i in 0..4 {
        unsafe { out.store_unchecked(i as u32, values[i]) };
    }
}

#[kernel]
pub unsafe fn float_bit_methods(out: WriteBuffer<u32>, a: f32, b: f32) {
    unsafe {
        out.store_unchecked(0, a.copysign(b).to_bits());
        out.store_unchecked(1, a.signum().to_bits());
        out.store_unchecked(2, a.total_cmp(&b) as i32 as u32);
    }
}

#[kernel]
pub unsafe fn float_clamp(out: WriteBuffer<f32>, value: f32, low: f32, high: f32) {
    unsafe { out.store_unchecked(0, value.clamp(low, high)) };
}

#[kernel]
pub unsafe fn unsigned_clamp(out: WriteBuffer<u32>, value: u32, low: u32, high: u32) {
    unsafe { out.store_unchecked(0, value.clamp(low, high)) };
}

#[kernel]
pub unsafe fn signed_clamp(out: WriteBuffer<i32>, value: i32, low: i32, high: i32) {
    unsafe { out.store_unchecked(0, value.clamp(low, high)) };
}

#[kernel]
pub unsafe fn checked_division(out: WriteBuffer<u32>, a: u32, b: u32, c: i32, d: i32) {
    let values = [
        a.checked_div(b).unwrap_or(0),
        a.checked_div(b).is_some() as u32,
        a.checked_rem(b).unwrap_or(0),
        a.checked_rem(b).is_some() as u32,
        c.checked_div(d).unwrap_or(0) as u32,
        c.checked_div(d).is_some() as u32,
        c.checked_rem(d).unwrap_or(0) as u32,
        c.checked_rem(d).is_some() as u32,
    ];
    for i in 0..8 {
        unsafe { out.store_unchecked(i as u32, values[i]) };
    }
}

#[kernel]
pub unsafe fn bit_methods(out: WriteBuffer<u64>, value: u64) {
    let nonzero = NonZeroU32::new(value as u32);
    let values = [
        (value as u8).swap_bytes() as u64,
        (value as i8).swap_bytes() as u64,
        (value as u16).swap_bytes() as u64,
        (value as i16).swap_bytes() as u64,
        (value as u32).swap_bytes() as u64,
        (value as i32).swap_bytes() as u64,
        value.swap_bytes(),
        (value as usize).swap_bytes() as u64,
        (value as isize).swap_bytes() as u64,
        nonzero.map(|n| n.leading_zeros()).unwrap_or(32) as u64,
        nonzero.map(|n| n.trailing_zeros()).unwrap_or(32) as u64,
    ];
    for i in 0..11 {
        unsafe { out.store_unchecked(i as u32, values[i]) };
    }
}
