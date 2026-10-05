#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

fn unit(out: WriteBuffer<u32>) {
    // SAFETY: the caller provides a writable element and launches one thread.
    unsafe { out.store_unchecked(0, 7) };
}

#[kernel]
pub unsafe fn unit_copy(out: WriteBuffer<u32>) {
    let u = unit(out);
    u
}
