#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

/// `y = a * x + y`: each thread reads and writes its own element of `y`.
#[kernel(requires(x.len() == y.len()))]
pub fn saxpy(a: f32, x: ReadBuffer<f32>, mut y: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(y) = y.get_mut(&index) {
        *y += a * x.as_slice()[index.get()];
    }
}
