#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

/// Adds two buffers element by element; threads past the end of a buffer write nothing.
///
/// # Safety
///
/// Nothing writes `a` or `b` while the kernel runs.
#[kernel]
pub unsafe fn vec_add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let a = a.as_slice();
    let b = b.as_slice();
    let i = index.get();
    if let Some(out) = out.get_mut(&index)
        && i < a.len()
        && i < b.len()
    {
        *out = a[i] + b[i];
    }
}
