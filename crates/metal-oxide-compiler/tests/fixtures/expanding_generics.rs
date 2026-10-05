#![no_std]
#![recursion_limit = "8"]
use metal_oxide_device::kernel;

fn expand<T>(value: u32) {
    if value > 0 {
        expand::<(T, T)>(value - 1);
    }
}

#[kernel]
pub unsafe fn invalid(value: u32) {
    expand::<u32>(value);
}
