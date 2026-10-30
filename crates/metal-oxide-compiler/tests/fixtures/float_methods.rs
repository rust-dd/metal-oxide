#![no_std]

use metal_oxide_device::{DisjointBuffer, Float, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn float_methods(input: ReadBuffer<(f32, f32, i32)>, mut output: DisjointBuffer<[f32; 10]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let (x, y, k) = input.as_slice()[index.get()];
        *output = [
            x.exp2(),
            x.log2(),
            x.log10(),
            x.tan(),
            x.asin(),
            x.acos(),
            x.atan2(y),
            x.rsqrt(),
            x.hypot(y),
            x.powi(k),
        ];
    }
}
