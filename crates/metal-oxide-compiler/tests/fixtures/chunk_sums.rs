#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

/// Writes, per thread, the sum and the pairwise products of its four input elements.
///
/// # Safety
/// The launch covers `out.len()` threads.
#[kernel]
pub unsafe fn chunk_sums(input: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let input = input.as_slice();
    if let Some(out) = out.get_mut(&index) {
        let start = index.get() * 4;
        let chunk = &input[start..start + 4];
        let mut total = 0.0;
        for value in chunk {
            total += *value;
        }
        for pair in chunk.chunks_exact(2) {
            total += pair[0] * pair[1];
        }
        *out = total;
    }
}
