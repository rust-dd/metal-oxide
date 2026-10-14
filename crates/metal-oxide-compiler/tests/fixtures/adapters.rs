#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, WriteBuffer, kernel, thread_index};

/// # Safety
/// Nothing writes `a` or `b` while the kernel runs.
#[kernel]
pub unsafe fn dot(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let (a, b) = (a.as_slice(), b.as_slice());
    let skip = index.get();
    if let Some(out) = out.get_mut(&index) {
        *out = a.iter().skip(skip).zip(b).map(|(x, y)| x * y).sum();
    }
}

/// # Safety
/// `out` holds ten values; one thread writes them. Nothing writes `values`.
#[kernel]
pub unsafe fn adapters(values: ReadBuffer<f32>, out: WriteBuffer<f32>, x: f32) {
    let values = values.as_slice();
    let results = [
        values.iter().zip(&values[1..]).map(|(a, b)| a * b).sum::<f32>(),
        values.iter().rev().fold(0.0, |acc, c| acc * x + c),
        values.iter().enumerate().map(|(i, v)| i as f32 * v).sum(),
        values.iter().filter(|v| **v > x).count() as f32,
        values.iter().position(|v| *v > x).map_or(-1.0, |p| p as f32),
        if values.iter().all(|v| *v > 0.0) { 1.0 } else { 0.0 },
        (0..values.len()).step_by(2).map(|i| values[i]).sum(),
        values.chunks_exact(2).map(|pair| pair[0] * pair[1]).sum(),
        values.iter().copied().fold(f32::MIN, f32::max),
        values.iter().skip(1).take(2).sum(),
    ];
    for (i, result) in results.iter().enumerate() {
        unsafe { out.store_unchecked(i as u32, *result) };
    }
}

/// # Safety
/// `out` holds four values; one thread writes them.
#[kernel]
pub unsafe fn mutation(out: WriteBuffer<f32>, factor: f32) {
    let mut values = [1.0_f32, 2.0, 3.0, 4.0];
    let offsets = [0.5_f32, 1.5, 2.5, 3.5];
    for (value, offset) in values.iter_mut().zip(offsets.iter()) {
        *value = *value * factor + offset;
    }
    for (i, value) in values.iter().rev().enumerate() {
        unsafe { out.store_unchecked(i as u32, *value) };
    }
}
