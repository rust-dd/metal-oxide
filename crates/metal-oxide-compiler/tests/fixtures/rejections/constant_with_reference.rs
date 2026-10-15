#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

const PAIR: (u32, &u32) = (1, &5);

#[kernel]
pub unsafe fn constant_with_reference(out: WriteBuffer<u32>) {
    let pair = PAIR;
    unsafe { out.store_unchecked(0, pair.0 + *pair.1) };
}
