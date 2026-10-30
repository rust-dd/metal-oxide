#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn logarithms(input: ReadBuffer<(u32, i32, u64)>, mut output: DisjointBuffer<[u32; 8]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let (a, b, c) = input.as_slice()[index.get()];
        *output = [
            a.ilog2(),
            a.ilog10(),
            a.ilog(3),
            (a as u16 | 1).ilog10(),
            b.ilog2(),
            b.ilog10(),
            c.ilog2(),
            c.ilog10(),
        ];
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn square_roots(input: ReadBuffer<(i32, i64)>, mut output: DisjointBuffer<[i64; 4]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let (a, b) = input.as_slice()[index.get()];
        *output = [
            a.isqrt() as i64,
            (a as i16 & i16::MAX).isqrt() as i64,
            b.isqrt(),
            a.unsigned_abs().isqrt() as i64,
        ];
    }
}
