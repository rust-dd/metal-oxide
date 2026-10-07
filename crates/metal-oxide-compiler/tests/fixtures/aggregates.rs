#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[derive(Clone, Copy)]
struct Sample {
    pair: (f32, f32),
    values: [f32; 3],
}

fn shift<const N: usize>(mut values: [f32; N]) -> [f32; N] {
    values[0] = values[1] + 2.0;
    values
}

#[kernel]
pub unsafe fn aggregates(out: WriteBuffer<f32>, seed: f32) {
    let mut value = Sample {
        pair: (seed, seed + 1.0),
        values: [seed; 3],
    };
    let original = value;
    value.values[1] = value.pair.1;
    let shifted = shift::<3>(value.values);
    unsafe {
        out.store_unchecked(0, original.values[1]);
        out.store_unchecked(1, value.values[1]);
        out.store_unchecked(2, shifted[0]);
        out.store_unchecked(3, shifted[1]);
        out.store_unchecked(4, shifted[2]);
    }
}
