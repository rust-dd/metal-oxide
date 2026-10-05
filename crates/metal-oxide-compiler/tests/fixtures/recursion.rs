#![no_std]
use metal_oxide_device::kernel;

fn recursive(value: u32) -> u32 {
    if value == 0 { 0 } else { recursive(value - 1) }
}

#[kernel]
pub unsafe fn invalid(value: u32) {
    let _ = recursive(value);
}
