#![no_std]
use metal_oxide_device::{F16, WriteBuffer, kernel, log, thread_idx};

#[cfg(not(any(mismatch, debug_format)))]
#[kernel]
pub unsafe fn logging(out: WriteBuffer<u32>, scale: f32, half: F16) {
    let i = thread_idx().x;
    log!("thread {} scale {} half {} wide {} signed {} {{braces}} 100%", i, scale, half, 1u64 << 40, -3i32);
    unsafe { out.store_unchecked(i, i) };
}

#[cfg(mismatch)]
#[kernel]
pub unsafe fn logging(value: u32) {
    log!("{} {}", value);
}

#[cfg(debug_format)]
#[kernel]
pub unsafe fn logging(value: u32) {
    log!("{:?}", value);
}
