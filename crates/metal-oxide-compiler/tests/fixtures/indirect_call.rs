#![no_std]
use metal_oxide_device::kernel;

fn indirect(function: fn(u32) -> u32, value: u32) -> u32 {
    function(value)
}

fn identity(value: u32) -> u32 {
    value
}

#[kernel]
pub unsafe fn invalid(value: u32) {
    let _ = indirect(identity, value);
}
