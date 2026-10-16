#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

fn classify(value: f32) -> (f32, bool) {
    (value * 2.0, value > 0.0)
}

#[kernel]
pub unsafe fn float_boolean_tuple(values: WriteBuffer<f32>, flags: WriteBuffer<u32>, value: f32) {
    let (scaled, positive) = classify(value);
    // SAFETY: the caller supplies one element in each output and launches one thread.
    unsafe {
        values.store_unchecked(0, scaled);
        flags.store_unchecked(0, u32::from(positive));
    }
}
