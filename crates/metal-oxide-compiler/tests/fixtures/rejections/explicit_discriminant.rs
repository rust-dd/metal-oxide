#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[derive(Clone, Copy)]
enum Level {
    Low = 1,
    High = 7,
}

#[kernel]
pub unsafe fn explicit_discriminant(out: WriteBuffer<u32>, value: u32) {
    let level = if value > 3 { Level::High } else { Level::Low };
    unsafe { out.store_unchecked(0, level as u32) };
}
