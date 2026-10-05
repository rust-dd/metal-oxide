#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid(value: bool) {
    let _ = value;
}
