#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};
use particle_device_math::{Config, Particle, advance};

/// # Safety
/// Buffers cover n initialized elements, do not overlap, and have one writer per output index.
#[kernel(block = (64, 1, 1))]
pub unsafe fn particle_update(
    input: ReadBuffer<Particle>,
    output: WriteBuffer<Particle>,
    config: Config,
    n: u32,
) {
    let i = block_idx()
        .x
        .wrapping_mul(block_dim().x)
        .wrapping_add(thread_idx().x);
    if i < n {
        // SAFETY: the launch contract provides initialized n-element buffers and unique writers.
        unsafe {
            output.store_unchecked(i, advance::<2>(input.load_unchecked(i), config));
        }
    }
}

/// # Safety
/// Buffers cover n initialized elements, do not overlap, and have one writer per output index.
#[kernel(block = (64, 1, 1))]
pub unsafe fn particle_step(
    input: ReadBuffer<Particle>,
    output: WriteBuffer<Particle>,
    config: Config,
    n: u32,
) {
    let i = block_idx()
        .x
        .wrapping_mul(block_dim().x)
        .wrapping_add(thread_idx().x);
    if i < n {
        // SAFETY: the launch contract provides initialized n-element buffers and unique writers.
        unsafe {
            output.store_unchecked(i, advance::<1>(input.load_unchecked(i), config));
        }
    }
}
