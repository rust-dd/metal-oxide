#![no_std]
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid<T>(value: T) {
    let _ = value;
}
