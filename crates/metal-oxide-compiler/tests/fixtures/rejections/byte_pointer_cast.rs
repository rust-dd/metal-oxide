#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn byte_pointer_cast(out: WriteBuffer<u32>, value: u32) {
    let pointer = &value as *const u32 as *const u8;
    unsafe { out.store_unchecked(0, u32::from(*pointer)) };
}
