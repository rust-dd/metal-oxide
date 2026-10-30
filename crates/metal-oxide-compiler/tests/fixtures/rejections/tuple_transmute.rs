#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn tuple_transmute(out: WriteBuffer<u32>, value: u32) {
    let halves = unsafe { core::mem::transmute::<_, (u16, u16)>(value) };
    unsafe { out.store_unchecked(0, u32::from(halves.0)) };
}
