#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn huge_array(out: WriteBuffer<u32>, value: u32) {
    let bytes = [0_u8; 1 << 32];
    unsafe { out.store_unchecked(0, u32::from(bytes[value as usize])) };
}
