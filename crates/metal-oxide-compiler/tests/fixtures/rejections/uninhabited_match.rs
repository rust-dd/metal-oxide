#![no_std]
use core::convert::Infallible;
use metal_oxide_device::kernel;

#[kernel]
pub unsafe fn invalid() {
    let never = unsafe { core::mem::transmute::<_, &Infallible>(&()) };
    match *never {}
}
