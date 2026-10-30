#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(a.len() == out.len(), b.len() == out.len()))]
pub fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        let i = index.get();
        *out = a.as_slice()[i] + b.as_slice()[i];
    }
}

#[kernel(block = (64, 1, 1), requires(2 * n + 1 <= values.len(), n > 0))]
pub fn pairs(values: ReadBuffer<f32>, mut out: DisjointBuffer<f32>, n: u32) {
    let index = thread_index();
    let values = values.as_slice();
    if let Some(out) = out.get_mut(&index) {
        let i = index.get() % n as usize;
        *out = values[2 * i] + values[2 * i + 1];
    }
}

#[kernel(requires(values.len() == out.len(), floor <= 0))]
pub fn clamp_below(values: ReadBuffer<i64>, mut out: DisjointBuffer<i64>, floor: i64) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        *out = values.as_slice()[index.get()].max(floor);
    }
}
