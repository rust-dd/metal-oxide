#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

pub union Slot<'a> {
    pub value: u32,
    pub pointer: &'a u32,
}

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let slot = Slot { value };
    unsafe { output.store_unchecked(0, slot.value) };
}
