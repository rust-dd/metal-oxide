#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

unsafe extern "Rust" {
    #[link_name = "__metal_sqrt_f32"]
    fn sqrt(value: f32) -> f32;
}

#[kernel]
pub unsafe fn foreign_alias(out: WriteBuffer<f32>, value: f32) {
    unsafe {
        out.store_unchecked(0, sqrt(value));
    }
}
