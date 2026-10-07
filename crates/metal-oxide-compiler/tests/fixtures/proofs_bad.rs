#![no_std]
#![allow(unused_assignments, unconditional_panic)]

use metal_oxide_device::{WriteBuffer, kernel};

#[cfg(join)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<i32>, value: i32, select: u32) {
    let divisor = if select == 0 { 4 } else { 0 };
    unsafe { out.store_unchecked(0, value / divisor); }
}

#[cfg(reassign)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<i32>, value: i32, mut divisor: i32) {
    if divisor == 0 { return; }
    divisor = 0;
    unsafe { out.store_unchecked(0, value / divisor); }
}

#[cfg(signed)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<i32>, value: i32, divisor: i32) {
    if divisor == 0 { return; }
    unsafe { out.store_unchecked(0, value % divisor); }
}

#[cfg(index)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<u32>, index: u32) {
    let values = [10_u32, 20, 30, 40];
    unsafe { out.store_unchecked(0, values[index as usize]); }
}

#[cfg(index_reassign)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<u32>, mut index: u32, other: u32) {
    let values = [10_u32, 20, 30, 40];
    if index < 4 {
        index = other;
        unsafe { out.store_unchecked(0, values[index as usize]); }
    }
}

#[cfg(stale_condition)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<u32>, mut index: u32, other: u32) {
    let values = [10_u32, 20, 30, 40];
    let condition = index < 4;
    index = other;
    if condition { unsafe { out.store_unchecked(0, values[index as usize]); } }
}

#[cfg(overflow)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<u32>, value: u32, other: u32) {
    if value < 16 { unsafe { out.store_unchecked(0, other + 1); } }
}

#[cfg(helper)]
fn quotient(value: i32, divisor: i32) -> i32 { value / divisor }

#[cfg(helper)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<i32>, value: i32, divisor: i32) {
    unsafe { out.store_unchecked(0, quotient(value, divisor)); }
}

#[cfg(truncated_index)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<u32>, high: u32) {
    let values = [10_u32, 20, 30, 40];
    let index = (high as usize) << 32;
    if (index as u32) < 4 { unsafe { out.store_unchecked(0, values[index]); } }
}

#[cfg(dynamic_store)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<u32>, index: u32) {
    let mut values = [10_u32, 20, 30, 40];
    if index < 4 {
        values[index as usize] = u32::MAX;
        unsafe { out.store_unchecked(0, values[0] + 1); }
    }
}

#[cfg(false_and)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<u32>, index: u32, other: u32) {
    let values = [10_u32, 20, 30, 40];
    if !((index < 4) & (other == 0)) && other != 0 {
        unsafe { out.store_unchecked(0, values[index as usize]); }
    }
}

#[cfg(division_range)]
#[kernel]
pub unsafe fn bad(out: WriteBuffer<i8>, divisor: i8) {
    if divisor < -10 || divisor > 10 || divisor == -1 || divisor == 0 { return; }
    let value = -100_i8 / divisor;
    unsafe { out.store_unchecked(0, value + 100); }
}
