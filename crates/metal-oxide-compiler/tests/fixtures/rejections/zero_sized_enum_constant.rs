#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

enum Mode {
    Only,
}

const MODE: Mode = Mode::Only;

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    let mode = MODE;
    unsafe { output.store_unchecked(0, mode as u32) };
}
