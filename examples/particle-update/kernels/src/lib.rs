#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};
use particle_device_math::{Config, Particle, advance};

/// Advances each particle by two steps; threads past the end of a buffer write nothing.
#[kernel(block = (64, 1, 1))]
pub fn particle_update(
    input: ReadBuffer<Particle>,
    mut output: DisjointBuffer<Particle>,
    config: Config,
) {
    let index = thread_index();
    let input = input.as_slice();
    let i = index.get();
    if let Some(output) = output.get_mut(&index)
        && i < input.len()
    {
        *output = advance::<2>(input[i], config);
    }
}

/// Advances each particle by one step; threads past the end of a buffer write nothing.
#[kernel(block = (64, 1, 1))]
pub fn particle_step(
    input: ReadBuffer<Particle>,
    mut output: DisjointBuffer<Particle>,
    config: Config,
) {
    let index = thread_index();
    let input = input.as_slice();
    let i = index.get();
    if let Some(output) = output.get_mut(&index)
        && i < input.len()
    {
        *output = advance::<1>(input[i], config);
    }
}
