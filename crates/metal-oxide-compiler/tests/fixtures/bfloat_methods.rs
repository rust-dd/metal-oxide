#![no_std]

use metal_oxide_device::{BF16, DisjointBuffer, Float, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn bfloat_methods(input: ReadBuffer<[BF16; 2]>, mut output: DisjointBuffer<[BF16; 12]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let [a, b] = input.as_slice()[index.get()];
        let (sine, cosine) = a.sin_cos();
        *output = [
            a.sqrt(),
            a.exp(),
            a.ln(),
            sine,
            cosine,
            a.atan2(b),
            a.hypot(b),
            a.powi(-3),
            a.rsqrt(),
            a.abs(),
            a.min(b),
            a.max(b),
        ];
    }
}
