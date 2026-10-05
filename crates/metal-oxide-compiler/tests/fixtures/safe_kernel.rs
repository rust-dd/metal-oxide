#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub fn invalid(value: u32) {
    let _ = value;
}
