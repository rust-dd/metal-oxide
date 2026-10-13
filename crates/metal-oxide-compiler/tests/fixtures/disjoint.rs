#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

/// # Safety
/// The launch covers at least `out.len()` threads.
#[kernel]
pub unsafe fn add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let a = a.as_slice();
    let b = b.as_slice();
    if let Some(out) = out.get_mut(&index) {
        let i = index.get();
        *out = a[i] + b[i];
    }
}
