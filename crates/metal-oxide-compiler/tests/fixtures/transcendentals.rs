#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, math, thread_idx};

#[kernel]
pub unsafe fn transcendentals(
    input: ReadBuffer<f32>,
    exponent: ReadBuffer<f32>,
    out: WriteBuffer<f32>,
    n: u32,
) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        unsafe {
            let x = input.load_unchecked(i);
            let y = exponent.load_unchecked(i);
            let base = i * 8;
            out.store_unchecked(base, math::ln(x));
            out.store_unchecked(base + 1, math::exp(x));
            out.store_unchecked(base + 2, math::sin(x));
            out.store_unchecked(base + 3, math::cos(x));
            out.store_unchecked(base + 4, math::powf(x, y));
            out.store_unchecked(base + 5, math::tanh(x));
            out.store_unchecked(base + 6, math::atan(x));
            out.store_unchecked(base + 7, math::floor(x));
        }
    }
}

fn normal(u: f32, v: f32) -> f32 {
    math::sqrt(-2.0 * math::ln(u)) * math::cos(core::f32::consts::TAU * v)
}

#[kernel]
pub unsafe fn box_muller_step(
    uniforms: ReadBuffer<f32>,
    angles: ReadBuffer<f32>,
    out: WriteBuffer<f32>,
    n: u32,
) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        unsafe {
            let z = normal(uniforms.load_unchecked(i), angles.load_unchecked(i));
            out.store_unchecked(i, 100.0 * math::exp(0.01 + 0.2 * z));
        }
    }
}
