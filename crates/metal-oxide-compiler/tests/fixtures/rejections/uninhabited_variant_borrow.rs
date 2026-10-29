#![no_std]
use core::convert::Infallible;
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    let pair = None::<(u32, Infallible)>;
    let value = match pair {
        Some(ref pair) => pair.0,
        None => 0,
    };
    unsafe { output.store_unchecked(0, value) };
}
