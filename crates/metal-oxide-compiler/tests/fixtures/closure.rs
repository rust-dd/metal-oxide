#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid(value: u32) {
    let closure = |x: u32| x + 1;
    let _ = closure(value);
}
