#![no_std]
use metal_oxide_device::{Dim3, WriteBuffer, kernel};

const ORIGIN: Option<Dim3> = None;

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    let x = match ORIGIN {
        Some(origin) => origin.x,
        None => 0,
    };
    unsafe { output.store_unchecked(0, x) };
}
