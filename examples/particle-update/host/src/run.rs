use crate::kernels::{self, Config, Motion, Particle};
use metal_oxide::{Argument, Device, LaunchConfig, Module, Pipeline};

pub(super) fn config() -> Config {
    Config {
        dt: 0.125,
        acceleration: (0.5, -0.25),
    }
}

pub(super) fn particles(len: usize) -> Vec<Particle> {
    (0..len)
        .map(|i| Particle {
            tag: (i % 3) as u8,
            flags: (i % 17) as u16,
            id: i as u32,
            motion: Motion {
                position: ((i % 256) as f32 * 0.25, (i % 128) as f32 * -0.5),
                velocity: [(i % 16) as f32 * 0.25, (i % 32) as f32 * -0.125],
            },
            color: [(i % 255) as u8, 19, 23],
        })
        .collect()
}

pub(super) fn expected(input: &[Particle], config: Config, steps: u16) -> Vec<Particle> {
    input
        .iter()
        .map(|original| {
            let mut value = *original;
            if value.tag == 0 {
                return value;
            }
            for _ in 0..steps {
                value.motion.velocity[0] += config.acceleration.0 * config.dt;
                value.motion.velocity[1] += config.acceleration.1 * config.dt;
                value.motion.position.0 += value.motion.velocity[0] * config.dt;
                value.motion.position.1 += value.motion.velocity[1] * config.dt;
            }
            value.flags = value.flags.wrapping_add(steps);
            value.color[1] = original.color[0];
            value
        })
        .collect()
}

pub(super) fn verify() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let module = Module::from_source(&device, include_str!("../../reference.metal"))?;
    let reference = Pipeline::new(&device, &module, "particle_reference")?;
    let sentinel = Particle {
        tag: 255,
        flags: u16::MAX,
        id: u32::MAX,
        motion: Motion {
            position: (-999.0, -999.0),
            velocity: [-999.0; 2],
        },
        color: [255; 3],
    };
    println!("Device: {}", device.name());
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let initial = particles(n as usize + 64);
        let input = device.buffer_from_slice(&initial)?;
        let mut output = device.buffer_from_slice(&vec![sentinel; initial.len()])?;
        let mut handwritten = device.buffer_from_slice(&vec![sentinel; initial.len()])?;
        let launch = LaunchConfig::<64>::for_elements(n)?;
        // SAFETY: disjoint initialized buffers cover n; padded threads take the guard branch.
        unsafe {
            kernels.particle_update(launch, &input, &mut output, config(), n)?;
            device.launch(
                &reference,
                launch,
                &[
                    Argument::read(&input),
                    Argument::write(&mut handwritten),
                    Argument::value(config())?,
                    Argument::value(n)?,
                ],
            )?;
        }
        let cpu = expected(&initial[..n as usize], config(), 2);
        assert_eq!(&output.as_slice()[..n as usize], cpu);
        assert_eq!(output.as_slice(), handwritten.as_slice());
        assert_eq!(&output.as_slice()[n as usize..], vec![sentinel; 64]);
        assert_eq!(input.as_slice(), initial);
        println!("particle-update n={n}: Rust, MSL, and CPU agree");
    }
    Ok(())
}

pub(super) fn verify_chain() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let initial = particles(257);
    let input = device.buffer_from_slice(&initial)?;
    let mut middle = device.buffer_zeroed::<Particle>(initial.len())?;
    let mut output = device.buffer_zeroed::<Particle>(initial.len())?;
    let launch = LaunchConfig::<64>::for_elements(initial.len() as u32)?;
    // SAFETY: the second kernel reads the first kernel's completed writes in the same batch.
    unsafe {
        device.submit(|batch| {
            kernels.enqueue_particle_update(batch, launch, &input, &mut middle, config(), 257)?;
            kernels.enqueue_particle_step(batch, launch, &middle, &mut output, config(), 257)
        })?
    }
    .wait()?;
    assert_eq!(middle.as_slice(), expected(&initial, config(), 2));
    assert_eq!(output.as_slice(), expected(&initial, config(), 3));
    Ok(())
}

#[cfg(test)]
pub(super) fn verify_cancelled() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let initial = particles(257);
    let input = device.buffer_from_slice(&initial)?;
    let launch = LaunchConfig::<64>::for_elements(257)?;
    for forget in [false, true] {
        let mut output = device.buffer_zeroed::<Particle>(257)?;
        // SAFETY: output has one writer per element; completion retains library/buffer/constant state.
        let submission = unsafe {
            device.submit(|batch| {
                kernels.enqueue_particle_update(batch, launch, &input, &mut output, config(), 257)
            })?
        };
        if forget {
            std::mem::forget(submission);
        } else {
            drop(submission);
        }
        assert_eq!(output.as_slice(), expected(&initial, config(), 2));
    }
    Ok(())
}
