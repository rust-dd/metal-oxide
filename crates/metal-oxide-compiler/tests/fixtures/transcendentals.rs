#![no_std]

use metal_oxide_device::{Float, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

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
            out.store_unchecked(base, x.ln());
            out.store_unchecked(base + 1, x.exp());
            out.store_unchecked(base + 2, x.sin());
            out.store_unchecked(base + 3, x.cos());
            out.store_unchecked(base + 4, x.powf(y));
            out.store_unchecked(base + 5, x.tanh());
            out.store_unchecked(base + 6, x.atan());
            out.store_unchecked(base + 7, x.floor());
        }
    }
}

fn normal(u: f32, v: f32) -> f32 {
    (-2.0 * u.ln()).sqrt() * (core::f32::consts::TAU * v).cos()
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
            out.store_unchecked(i, 100.0 * (0.01 + 0.2 * z).exp());
        }
    }
}
