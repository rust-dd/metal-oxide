#![no_std]

#[derive(Clone, Copy)]
pub struct Particle {
    pub position: (f32, f32),
    pub velocity: [f32; 2],
    pub tag: u8,
}

#[derive(Clone, Copy)]
pub struct Config {
    pub dt: f32,
}

pub fn advance<const STEPS: u32>(mut particle: Particle, config: Config) -> Particle {
    if particle.tag == 0 {
        return particle;
    }
    let mut step = 0_u32;
    while step < STEPS {
        particle.position.0 += particle.velocity[0] * config.dt;
        particle.position.1 += particle.velocity[1] * config.dt;
        step = step.wrapping_add(1);
    }
    particle
}
