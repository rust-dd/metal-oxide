#![no_std]
use core::convert::Infallible;
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    let pair: Option<(u32, Infallible)> = None;
    let value = match pair {
        Some(ref pair) => pair.0,
        None => 0,
    };
    unsafe { output.store_unchecked(0, value) };
}
