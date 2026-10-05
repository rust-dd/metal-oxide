#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid(value: u32) {
    let _: f32 = value;
}
