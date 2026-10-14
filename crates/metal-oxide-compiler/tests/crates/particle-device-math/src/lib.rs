#![no_std]

#[derive(Clone, Copy)]
pub struct Motion {
    pub position: (f32, f32),
    pub velocity: [f32; 2],
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub tag: u8,
    pub flags: u16,
    pub id: u32,
    pub motion: Motion,
    pub color: [u8; 3],
}

#[derive(Clone, Copy)]
pub struct Config {
    pub dt: f32,
    pub acceleration: (f32, f32),
}

pub fn advance<const STEPS: u32>(mut particle: Particle, config: Config) -> Particle {
    let original = particle;
    if particle.tag == 0 {
        return original;
    }
    let mut step = 0_u32;
    while step < STEPS {
        let before = particle.motion;
        particle.motion.velocity[0] = before.velocity[0] + config.acceleration.0 * config.dt;
        particle.motion.velocity[1] = before.velocity[1] + config.acceleration.1 * config.dt;
        particle.motion.position.0 += particle.motion.velocity[0] * config.dt;
        particle.motion.position.1 += particle.motion.velocity[1] * config.dt;
        step = step.wrapping_add(1);
    }
    particle.flags = particle.flags.wrapping_add(STEPS as u16);
    particle.color[1] = original.color[0];
    particle
}
