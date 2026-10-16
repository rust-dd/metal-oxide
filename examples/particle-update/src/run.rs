use metal_oxide::{Argument, Device, Launch, LaunchConfig, Module, Operation, Pipeline, sequence};
use particle_update::particles::{self, Config, Motion, Particle, advance};

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

/// The particles after `steps` steps, advanced on the CPU by the code the kernels run.
pub(super) fn expected(input: &[Particle], config: Config, steps: u16) -> Vec<Particle> {
    input
        .iter()
        .map(|particle| match steps {
            1 => advance::<1>(*particle, config),
            2 => advance::<2>(*particle, config),
            3 => advance::<1>(advance::<2>(*particle, config), config),
            _ => unreachable!("the examples advance up to three steps"),
        })
        .collect()
}

pub(super) fn verify() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, include_str!("../reference.metal"))?;
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
        let initial = particles(n as usize);
        let input = device.buffer_from_slice(&initial)?;
        let mut output = device.buffer_from_slice(&vec![sentinel; initial.len() + 64])?;
        let mut handwritten = device.buffer_from_slice(&vec![sentinel; initial.len() + 64])?;
        let launch = LaunchConfig::<64>::for_elements(n)?;
        particles::particle_update(&device, launch, &input, &mut output, config()).wait()?;
        // SAFETY: the handwritten kernel stops at n, which the input covers.
        unsafe {
            Launch::new(
                Ok(&reference),
                launch,
                (&input, &mut handwritten, config(), n),
                |(input, handwritten, config, n)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(handwritten),
                        Argument::value(*config)?,
                        Argument::value(*n)?,
                    ])
                },
            )
            .wait()?;
        }
        let cpu = expected(&initial, config(), 2);
        assert_eq!(&output.read().wait()[..n as usize], cpu);
        assert_eq!(output.read().wait(), handwritten.read().wait());
        assert_eq!(&output.read().wait()[n as usize..], vec![sentinel; 64]);
        assert_eq!(input.read().wait(), initial);
        println!("particle-update n={n}: Rust, MSL, and CPU agree");
    }
    Ok(())
}

pub(super) fn verify_chain() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let initial = particles(257);
    let input = device.buffer_from_slice(&initial)?;
    let mut middle = device.buffer_zeroed::<Particle>(initial.len())?;
    let mut output = device.buffer_zeroed::<Particle>(initial.len())?;
    let launch = LaunchConfig::<64>::for_elements(initial.len() as u32)?;
    sequence(|batch| {
        batch.push(particles::particle_update(
            &device,
            launch,
            &input,
            &mut middle,
            config(),
        ))?;
        batch.push(particles::particle_step(
            &device,
            launch,
            &middle,
            &mut output,
            config(),
        ))?;
        Ok(())
    })
    .wait()?;
    assert_eq!(middle.read().wait(), expected(&initial, config(), 2));
    assert_eq!(output.read().wait(), expected(&initial, config(), 3));
    Ok(())
}

#[cfg(test)]
pub(super) fn verify_cancelled() -> metal_oxide::Result<()> {
    use std::{
        future::{Future, IntoFuture},
        pin::Pin,
        task::{Context, Waker},
    };
    let device = Device::system_default()?;
    let initial = particles(257);
    let input = device.buffer_from_slice(&initial)?;
    let launch = LaunchConfig::<64>::for_elements(257)?;
    // Warm the pipeline so each cancelled launch submits on its first poll.
    let mut warmup = device.buffer_zeroed::<Particle>(257)?;
    particles::particle_update(&device, launch, &input, &mut warmup, config()).wait()?;
    drop(warmup);
    for forget in [false, true] {
        let mut output = device.buffer_zeroed::<Particle>(257)?;
        let operation = particles::particle_update(&device, launch, &input, &mut output, config());
        let mut future = operation.into_future();
        let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
        if forget {
            std::mem::forget(future);
        } else {
            drop(future);
        }
        assert_eq!(output.read().wait(), expected(&initial, config(), 2));
    }
    Ok(())
}
