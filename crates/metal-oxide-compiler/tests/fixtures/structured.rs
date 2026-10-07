#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[derive(Clone, Copy)]
pub struct Particle {
    pub tag: u8,
    pub flags: u16,
    pub id: u32,
    pub position: (f32, f32),
    pub velocity: [f32; 2],
}

#[derive(Clone, Copy)]
pub struct Config {
    pub dt: f32,
    pub acceleration: (f32, f32),
}

fn step(mut value: Particle, config: Config) -> Particle {
    value.velocity[0] += config.acceleration.0 * config.dt;
    value.velocity[1] += config.acceleration.1 * config.dt;
    value.position.0 += value.velocity[0] * config.dt;
    value.position.1 += value.velocity[1] * config.dt;
    value
}

#[kernel]
pub unsafe fn structured(input: ReadBuffer<Particle>, out: WriteBuffer<Particle>, config: Config, n: u32) {
    let i = block_idx().x.wrapping_mul(block_dim().x).wrapping_add(thread_idx().x);
    if i < n {
        unsafe { out.store_unchecked(i, step(input.load_unchecked(i), config)); }
    }
}
