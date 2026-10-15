#![no_std]
use metal_oxide_device::kernel;

pub struct Guard(pub u32);
impl Drop for Guard {
    fn drop(&mut self) {}
}

#[kernel]
pub unsafe fn invalid(value: u32) {
    core::mem::forget(Guard(value));
}
