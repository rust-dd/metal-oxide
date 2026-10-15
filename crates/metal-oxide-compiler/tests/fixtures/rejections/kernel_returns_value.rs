#![no_std]

use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn returns_value() -> u32 {
    0
}
