#![no_std]
use core::convert::Infallible;
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid() {
    let never: &Infallible = unsafe { core::mem::transmute(&()) };
    match *never {}
}
