#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

fn sum(values: &[f32]) -> f32 {
    let mut total = 0.0;
    for value in values {
        total += *value;
    }
    total
}

fn scale(values: &mut [f32], factor: f32) {
    for value in values.iter_mut() {
        *value *= factor;
    }
}

fn pairs(values: &[f32]) -> f32 {
    let mut total = 0.0;
    for pair in values.chunks_exact(2) {
        total += pair[0] * pair[1];
    }
    total
}

/// # Safety
/// `out` holds three values; one thread writes them.
#[kernel]
pub unsafe fn slice_iterators(out: WriteBuffer<f32>, factor: f32) {
    let mut values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    scale(&mut values, factor);
    unsafe { out.store_unchecked(0, sum(&values)) };
    unsafe { out.store_unchecked(1, pairs(&values)) };
    unsafe { out.store_unchecked(2, sum(&values[1..]) - pairs(&values[1..])) };
}
