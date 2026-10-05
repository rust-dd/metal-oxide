#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid(mut value: u32) {
    let first = &mut value;
    let second = &mut value;
    *first += *second;
}
