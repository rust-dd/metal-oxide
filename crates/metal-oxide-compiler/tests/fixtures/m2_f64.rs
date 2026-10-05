#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn wide(out: WriteBuffer<f32>, n: u32) {
    let value = n as f64 + 1.0;
    unsafe { out.store_unchecked(0, value as f32); }
}
