#![no_std]
use metal_oxide_device::kernel;

unsafe extern "C" {
    fn foreign(value: u32) -> u32;
}

#[kernel]
pub unsafe fn invalid(value: u32) {
    // SAFETY: this fixture is analyzed and never launched.
    let _ = unsafe { foreign(value) };
}
