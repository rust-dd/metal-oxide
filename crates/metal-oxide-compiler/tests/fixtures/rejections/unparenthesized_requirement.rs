#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
#[metal_oxide_compiler::requires = "n > 0"]
pub unsafe fn written_requirement(out: WriteBuffer<u32>, n: u32) {
    unsafe { out.store_unchecked(0, n) };
}
