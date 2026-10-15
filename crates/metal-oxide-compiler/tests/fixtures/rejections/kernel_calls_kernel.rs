#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid(value: u32) {
    unsafe { callee(value) };
}

// Its caller comes first, so the call is imported before this entry.
#[kernel]
pub unsafe fn callee(_value: u32) {}
