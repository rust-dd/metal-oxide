#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid(value: &[f32]) {
    let _ = value;
}
