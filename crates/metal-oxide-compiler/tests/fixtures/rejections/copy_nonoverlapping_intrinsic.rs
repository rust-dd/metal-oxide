#![no_std]
#![allow(deprecated)]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let mut copy = 0;
    // The `intrinsics` path names the intrinsic itself rather than `core::ptr`'s wrapper.
    unsafe { core::intrinsics::copy_nonoverlapping(&value, &mut copy, 1) };
    unsafe { output.store_unchecked(0, copy) };
}
