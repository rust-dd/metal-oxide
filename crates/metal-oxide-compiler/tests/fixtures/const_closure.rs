#![no_std]
#![feature(const_closures, const_trait_impl)]
use metal_oxide_device::kernel;

const N: u32 = (const |value: u32| value + 1)(1);

#[kernel]
pub unsafe fn constant(value: u32) {
    let _ = value + N;
}
