use super::*;
use metal_oxide::{Argument, LaunchConfig};
use particle_update::particles::{Config as ParticleConfig, Motion, Particle, advance};

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
            .map(|particle| advance::<2>(*particle, parameters))
            .collect::<Vec<_>>();
        let cpu = elapsed(start);
        let start = Instant::now();
        input.write().wait().copy_from_slice(&values);
        let copy = elapsed(start);
        let start = Instant::now();
        input.upload();
        output.upload();
        let upload = elapsed(start);
        let mut sample = dispatch(|batch| {
            // SAFETY: the records' codecs match the shared layouts; the generated kernel stops at
            // the buffer lengths, the handwritten one at n, which both distinct buffers cover.
            unsafe {
                if variant == "rust" {
                    batch
                        .push(Launch::new(
                            Ok(&pipelines[0]),
                            launch,
                            (&input, &mut output, parameters),
                            |(input, output, parameters)| {
                                Ok([
                                    Argument::read(input),
                                    Argument::write(output),
                                    Argument::value(*parameters)?,
                                ])
                            },
                        ))
                        .map(drop)
                } else {
                    batch
                        .push(Launch::new(
                            Ok(&pipelines[0]),
                            launch,
                            (&input, &mut output, parameters, n),
                            |(input, output, parameters, n)| {
                                Ok([
                                    Argument::read(input),
                                    Argument::write(output),
                                    Argument::value(*parameters)?,
                                    Argument::value(*n)?,
                                ])
                            },
                        ))
                        .map(drop)
                }
            }
        })?;
        let start = Instant::now();
        let result = output.read().wait();
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
