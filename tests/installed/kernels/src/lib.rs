#![no_std]

mod scan;

use external_helper::{Config, Particle, advance};
use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, math, thread_idx};

/// # Safety
/// Disjoint buffers cover n elements; the launch is one-dimensional.
#[kernel(block = (64, 1, 1))]
pub unsafe fn update(
    input: ReadBuffer<Particle>,
    output: WriteBuffer<Particle>,
    config: Config,
    n: u32,
) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: the guarded index names one initialized input and one unique output.
        unsafe {
            output.store_unchecked(i, advance::<2>(input.load_unchecked(i), config));
        }
    }
}

/// # Safety
/// Disjoint buffers cover n elements; the launch is one-dimensional.
#[kernel(block = (64, 1, 1))]
pub unsafe fn select(input: ReadBuffer<Particle>, flags: WriteBuffer<u32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {
        // SAFETY: each valid input supplies one uniquely owned flag.
        unsafe {
            let x = input.load_unchecked(i).position.0;
            let magnitude = math::sqrt(math::fma(x, x, 0.0));
            flags.store_unchecked(i, (magnitude as u32) & 1);
        }
    }
}

/// # Safety
/// Disjoint buffers cover n elements. Flags are in {0,1}, prefix is their
/// exclusive scan, and the launch is one-dimensional.
#[kernel(block = (64, 1, 1))]
pub unsafe fn scatter(
    input: ReadBuffer<Particle>,
    flags: ReadBuffer<u32>,
    prefix: ReadBuffer<u32>,
    output: WriteBuffer<Particle>,
    n: u32,
) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    // SAFETY: selected elements have distinct prefix positions below n.
    unsafe {
        if i < n && flags.load_unchecked(i) != 0 {
            output.store_unchecked(prefix.load_unchecked(i), input.load_unchecked(i));
        }
    }
}
