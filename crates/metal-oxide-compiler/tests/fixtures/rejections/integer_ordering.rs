#![no_std]
#![feature(core_intrinsics)]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn rejected(output: WriteBuffer<i32>, left: u32, right: u32) {
    let ordering = core::intrinsics::three_way_compare(left, right);
    // SAFETY: the caller supplies one output element and launches one thread.
    unsafe { output.store_unchecked(0, ordering as i32) };
}
