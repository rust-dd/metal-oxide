use super::*;
use metal_oxide::{Argument, Buffer, LaunchConfig};
use particle_update::particles::{Config as ParticleConfig, Motion, Particle, advance};

pub(super) struct ParticleUpdate {
    case: Case,
    pipeline: Pipeline,
    n: u32,
    parameters: ParticleConfig,
    values: Vec<Particle>,
    launch: LaunchConfig<64>,
    input: Buffer<Particle>,
    output: Buffer<Particle>,
}

impl ParticleUpdate {
    pub(super) fn new(device: &Device, config: &Config, variant: &str) -> Result<Self> {
        let n = config.elements;
        let mut case = Case::new("particle-update", variant, vec![n]);
        let entry = if variant == "rust" {
            "particle_update"
        } else {
            "particle_reference"
        };
        let pipeline = case
            .pipelines(device, config, "particle-update", &[entry])?
            .remove(0);
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
        let input = device.buffer_zeroed::<Particle>(n as usize)?;
        let output = device.buffer_zeroed::<Particle>(n as usize)?;
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipeline,
            n,
            parameters,
            values,
            launch: LaunchConfig::<64>::for_elements(n)?,
            input,
            output,
        })
    }
}

impl Candidate for ParticleUpdate {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let (input, output) = (&mut self.input, &mut self.output);
        let (parameters, n) = (self.parameters, self.n);
        let start = Instant::now();
        let expected = self
            .values
            .iter()
            .map(|particle| advance::<2>(*particle, parameters))
            .collect::<Vec<_>>();
        let cpu = elapsed(start);
        let start = Instant::now();
        input.write().wait().copy_from_slice(&self.values);
        let copy = elapsed(start);
        let start = Instant::now();
        input.upload();
        output.upload();
        let upload = elapsed(start);
        let (pipeline, launch, rust) = (&self.pipeline, self.launch, self.case.variant == "rust");
        let mut sample = dispatch(|batch| {
            // SAFETY: the records' codecs match the shared layouts; the generated kernel stops at
            // the buffer lengths, the handwritten one at n, which both distinct buffers cover.
            unsafe {
                if rust {
                    batch
                        .push(Launch::new(
                            Ok(pipeline),
                            launch,
                            (&*input, &mut *output, parameters),
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
                            Ok(pipeline),
                            launch,
                            (&*input, &mut *output, parameters, n),
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
        let case = &mut self.case;
        assert_eq!(result, expected, "{} {}", case.name, case.variant);
        sample.cpu_reference_ns = Some(cpu);
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if keep {
            case.samples.push(sample);
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<Case> {
        self.case.finish()
    }
}
