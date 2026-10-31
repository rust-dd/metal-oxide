#![no_std]
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]
use metal_oxide_device::{WriteBuffer, kernel};

fn double(value: u32) -> u32 {
    value * 2
}

fn tail(value: u32) -> u32 {
    become double(value)
}

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    unsafe { output.store_unchecked(0, tail(value)) };
}
