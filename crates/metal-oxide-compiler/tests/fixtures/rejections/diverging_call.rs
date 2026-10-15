#![no_std]
use metal_oxide_device::kernel;

fn halt() -> ! {
    loop {}
}

#[kernel]
pub unsafe fn invalid(value: u32) {
    if value == 0 {
        halt();
    }
}
