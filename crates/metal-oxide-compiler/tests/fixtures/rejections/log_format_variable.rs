#![no_std]
use metal_oxide_device::{kernel, log};

#[kernel]
pub unsafe fn invalid(flag: u32) {
    let format = if flag == 0 { "zero" } else { "other" };
    log::write(format, ());
}
