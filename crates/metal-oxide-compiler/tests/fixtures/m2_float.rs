#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn float_math(out: WriteBuffer<f32>, a: f32, b: f32, c: f32) {
    let product = a * b;
    // SAFETY: the caller supplies a writable element and launches one thread.
    unsafe { out.store_unchecked(0, product + c) };
}
