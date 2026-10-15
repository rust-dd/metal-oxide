#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn array_transmute(out: WriteBuffer<u32>, value: u32) {
    let bytes: [u8; 4] = unsafe { core::mem::transmute(value) };
    unsafe { out.store_unchecked(0, u32::from(bytes[0])) };
}
