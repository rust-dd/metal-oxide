#![no_std]

use metal_oxide_device::kernel;

#[kernel]
pub unsafe extern "C" fn c_abi() {}
