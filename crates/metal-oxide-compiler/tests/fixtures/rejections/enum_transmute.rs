#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[repr(u32)]
#[derive(Clone, Copy)]
enum Mode {
    Off,
    On,
}

#[kernel]
pub unsafe fn enum_transmute(out: WriteBuffer<u32>, value: u32) {
    let mode = unsafe { core::mem::transmute::<_, Mode>(value & 1) };
    unsafe { out.store_unchecked(0, mode as u32) };
}
