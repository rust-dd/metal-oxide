#![no_std]
#![feature(f16)]

use metal_oxide_device::{BF16, F16, DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn f16_arithmetic(input: ReadBuffer<[F16; 2]>, mut output: DisjointBuffer<[F16; 8]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let [a, b] = input.as_slice()[index.get()];
        *output = [a + b, a - b, a * b, a / b, a % b, -a,
            F16::from_f32(if a < b { 1.0 } else { 0.0 }),
            F16::from_f32(if a == b { 1.0 } else { 0.0 })];
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn bf16_arithmetic(input: ReadBuffer<[BF16; 2]>, mut output: DisjointBuffer<[BF16; 8]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let [a, b] = input.as_slice()[index.get()];
        *output = [a + b, a - b, a * b, a / b, a % b, -a,
            BF16::from_f32(if a < b { 1.0 } else { 0.0 }),
            BF16::from_f32(if a == b { 1.0 } else { 0.0 })];
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn primitive(input: ReadBuffer<[f16; 2]>, mut output: DisjointBuffer<[f16; 8]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let [a, b] = input.as_slice()[index.get()];
        *output = [a + b, a - b, a * b, a / b, a % b, -a,
            if a < b { 1.0 } else { 0.0 }, if a == b { 1.0 } else { 0.0 }];
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn bfloat_round(input: ReadBuffer<f32>, mut output: DisjointBuffer<BF16>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = BF16::from_f32(input.as_slice()[index.get()]);
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn bfloat_widen(input: ReadBuffer<BF16>, mut output: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = input.as_slice()[index.get()].to_f32();
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn half_casts(input: ReadBuffer<f16>, mut output: DisjointBuffer<(u64, i32, f16)>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let value = input.as_slice()[index.get()];
        let integer = value as u64;
        *output = (integer, value as i32, integer as f16);
    }
}
