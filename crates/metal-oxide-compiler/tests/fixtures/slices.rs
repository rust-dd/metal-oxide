#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

fn sum(values: &[f32]) -> f32 {
    let mut total = 0.0;
    let mut i = 0;
    while i < values.len() {
        total += values[i];
        i += 1;
    }
    total
}

fn sum_range(values: &[f32]) -> f32 {
    let mut total = 0.0;
    for i in 0..values.len() {
        total += values[i];
    }
    total
}

fn scale(values: &mut [f32], factor: f32) {
    for i in 0..values.len() {
        values[i] *= factor;
    }
}

fn largest(values: &[f32; 4]) -> &f32 {
    let mut best = &values[0];
    if values[1] > *best {
        best = &values[1];
    }
    if values[2] > *best {
        best = &values[2];
    }
    if values[3] > *best {
        best = &values[3];
    }
    best
}

fn bump(value: &mut f32) {
    *value += 1.0;
}

/// # Safety
/// `out` holds eight values; one thread writes them.
#[kernel]
pub unsafe fn slices(out: WriteBuffer<f32>, seed: f32) {
    let mut values = [seed, seed + 1.0, seed + 2.0, seed + 3.0];
    bump(&mut values[3]);
    let all: &[f32] = &values;
    let results = [
        sum(all),
        sum_range(&all[1..3]),
        sum(&all[2..]),
        sum_range(&all[..2]),
        sum(&all[..]),
        *largest(&values),
    ];
    let mut scaled = values;
    scale(&mut scaled[1..], 2.0);
    let mut i = 0;
    while i < results.len() {
        unsafe { out.store_unchecked(i as u32, results[i]) };
        i += 1;
    }
    unsafe { out.store_unchecked(6, scaled[0]) };
    unsafe { out.store_unchecked(7, scaled[3]) };
}
