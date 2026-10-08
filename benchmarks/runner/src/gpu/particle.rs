use super::*;
use metal_oxide::{Argument, LaunchConfig};
use records::{Config as ParticleConfig, Motion, Particle};

pub(super) fn run(device: &Device, config: &Config, variant: &str) -> Result<Case> {
    let n = config.elements;
    let mut case = Case::new("particle-update", variant, vec![n]);
    let entry = if variant == "rust" {
        "particle_update"
    } else {
        "particle_reference"
    };
    let pipelines = case.pipelines(device, config, "particle-update", &[entry])?;
    let parameters = ParticleConfig {
        dt: 0.125,
        acceleration: (0.5, -0.25),
    };
    let values = (0..n)
        .map(|i| Particle {
            tag: (i % 3) as u8,
            flags: (i % 17) as u16,
            id: i,
            motion: Motion {
                position: ((i % 256) as f32 * 0.25, (i % 128) as f32 * -0.5),
                velocity: [(i % 16) as f32 * 0.25, (i % 32) as f32 * -0.125],
            },
            color: [(i % 255) as u8, 19, 23],
        })
        .collect::<Vec<_>>();
    let start = Instant::now();
    let mut input = device.buffer_zeroed::<Particle>(n as usize)?;
    let mut output = device.buffer_zeroed::<Particle>(n as usize)?;
    case.allocation_ns = elapsed(start);
    let launch = LaunchConfig::<64>::for_elements(n)?;
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        let expected = values
            .iter()
            .map(|original| {
                let mut value = *original;
                if value.tag != 0 {
                    for _ in 0..2 {
                        value.motion.velocity[0] += parameters.acceleration.0 * parameters.dt;
                        value.motion.velocity[1] += parameters.acceleration.1 * parameters.dt;
                        value.motion.position.0 += value.motion.velocity[0] * parameters.dt;
                        value.motion.position.1 += value.motion.velocity[1] * parameters.dt;
                    }
                    value.flags = value.flags.wrapping_add(2);
                    value.color[1] = original.color[0];
                }
                value
            })
            .collect::<Vec<_>>();
        let cpu = elapsed(start);
        let start = Instant::now();
        input.as_mut_slice().copy_from_slice(&values);
        let copy = elapsed(start);
        let start = Instant::now();
        input.upload();
        output.upload();
        let upload = elapsed(start);
        // SAFETY: generated codecs match the shared layouts; disjoint buffers cover n guarded threads.
        let mut sample = unsafe {
            dispatch(device, |batch| {
                batch.launch(
                    &pipelines[0],
                    launch,
                    &[
                        Argument::read(&input),
                        Argument::write(&mut output),
                        Argument::value(parameters)?,
                        Argument::value(n)?,
                    ],
                )
            })?
        };
        let start = Instant::now();
        let result = output.as_slice();
        sample.readback_ns = elapsed(start);
        assert_eq!(result, expected, "particle-update {variant}");
        sample.cpu_reference_ns = cpu;
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    case.finish()
}
