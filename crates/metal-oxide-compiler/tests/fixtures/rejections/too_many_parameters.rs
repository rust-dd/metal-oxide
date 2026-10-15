#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid(
    _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32,
    _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32,
    _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32,
) {
}
