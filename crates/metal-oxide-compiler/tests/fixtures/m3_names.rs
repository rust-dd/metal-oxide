#![no_std]

use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn names(_α: u32, r#type: f32, _: i32) {}
