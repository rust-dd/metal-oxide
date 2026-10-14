//! Particle records and the kernels that advance them, which the example and the benchmarks run.

/// Particles that the host creates and checks and its kernels advance, in one definition.
#[metal_oxide::kernels]
pub mod particles {
    use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Motion {
        pub position: (f32, f32),
        pub velocity: [f32; 2],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Particle {
        pub tag: u8,
        pub flags: u16,
        pub id: u32,
        pub motion: Motion,
        pub color: [u8; 3],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Config {
        pub dt: f32,
        pub acceleration: (f32, f32),
    }

    /// Advances a particle by `STEPS` steps; particles tagged 0 stay as they are.
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
}
