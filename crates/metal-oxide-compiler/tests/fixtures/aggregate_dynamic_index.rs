#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn dynamic_index(out: WriteBuffer<f32>, index: u32, seed: f32) {
    let values = [seed; 3];
    unsafe { out.store_unchecked(0, values[index as usize]); }
}
