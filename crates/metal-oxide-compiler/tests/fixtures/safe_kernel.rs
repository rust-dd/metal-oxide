#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel]
pub fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let (a, b) = (a.as_slice(), b.as_slice());
    let i = index.get();
    if let Some(out) = out.get_mut(&index)
        && i < a.len()
        && i < b.len()
    {
        *out = a[i] + b[i];
    }
}
