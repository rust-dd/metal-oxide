#![no_std]
use metal_oxide_device::{kernel, log};

#[kernel]
pub unsafe fn invalid(value: u32) {
    // `log!` expands to this call, but its error would point into the device crate.
    log::write("{:?}", (value,));
}
