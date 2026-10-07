#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

fn empty(_: [f32; 0]) -> f32 { 1.0 }

#[kernel]
pub unsafe fn empty_array(out: WriteBuffer<f32>) {
    unsafe { out.store_unchecked(0, empty([])); }
}
