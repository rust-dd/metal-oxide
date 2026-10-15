#![no_std]
use core::ops::Index;
use metal_oxide_device::{WriteBuffer, kernel};

fn first<C: Index<usize, Output = u32>>(values: &C) -> u32 {
    values[0]
}

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    unsafe { output.store_unchecked(0, first(&[value, value])) };
}
