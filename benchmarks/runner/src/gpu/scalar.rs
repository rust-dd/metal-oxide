use super::*;
use metal_oxide::{Argument, Buffer, LaunchConfig};

pub(super) struct Floats {
    case: Case,
    pipeline: Pipeline,
    reduction: bool,
    n: u32,
    values: Vec<f32>,
    launch: LaunchConfig<256>,
    buffers: (Buffer<f32>, Buffer<f32>, Buffer<f32>),
}

impl Floats {
    pub(super) fn new(
        device: &Device,
        config: &Config,
        variant: &str,
        reduction: bool,
    ) -> Result<Self> {
        let n = config.elements;
        let name = if reduction { "reduction" } else { "vec-add" };
        let mut case = Case::new(name, variant, vec![n]);
        let entry = if reduction { "reduce" } else { "vec_add" };
        let pipeline = case.pipelines(device, config, name, &[entry])?.remove(0);
        let values = (0..n).map(|i| (i % 31) as f32 * 0.25).collect::<Vec<_>>();
        let start = Instant::now();
        let a = device.buffer_zeroed::<f32>(n as usize)?;
        let b = device.buffer_zeroed::<f32>(if reduction { 0 } else { n } as usize)?;
        let out =
            device.buffer_zeroed::<f32>(if reduction { n.div_ceil(256) } else { n } as usize)?;
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipeline,
            reduction,
            n,
            values,
            launch: LaunchConfig::<256>::for_elements(n)?,
            buffers: (a, b, out),
        })
    }
}

impl Candidate for Floats {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let (n, reduction, rust) = (self.n, self.reduction, self.case.variant == "rust");
        let (a, b, out) = &mut self.buffers;
        let start = Instant::now();
        let expected = if reduction {
            self.values
                .chunks(256)
                .map(|chunk| chunk.iter().sum::<f32>())
                .collect::<Vec<_>>()
        } else {
            self.values.iter().map(|x| x + x).collect()
        };
        let cpu = elapsed(start);
        let start = Instant::now();
        a.write().wait().copy_from_slice(&self.values);
        if !reduction {
            b.write().wait().copy_from_slice(&self.values);
        }
        let copy = elapsed(start);
        let start = Instant::now();
        a.upload();
        b.upload();
        out.upload();
        let upload = elapsed(start);
        let (pipeline, launch) = (&self.pipeline, self.launch);
        let mut sample = dispatch(|batch| {
            // SAFETY: distinct buffers; generated kernels stop at the buffer lengths, handwritten
            // ones at n, which every buffer covers. All 256 lanes of a reduction block participate.
            unsafe {
                match (reduction, rust) {
                    (true, true) => batch
                        .push(Launch::new(
                            Ok(pipeline),
                            launch,
                            (&*a, &mut *out),
                            |(a, out)| Ok([Argument::read(a), Argument::write(out)]),
                        ))
                        .map(drop),
                    (true, false) => batch
                        .push(Launch::new(
                            Ok(pipeline),
                            launch,
                            (&*a, &mut *out, n),
                            |(a, out, n)| {
                                Ok([
                                    Argument::read(a),
                                    Argument::write(out),
                                    Argument::value(*n)?,
                                ])
                            },
                        ))
                        .map(drop),
                    (false, true) => batch
                        .push(Launch::new(
                            Ok(pipeline),
                            launch,
                            (&*a, &*b, &mut *out),
                            |(a, b, out)| {
                                Ok([Argument::read(a), Argument::read(b), Argument::write(out)])
                            },
                        ))
                        .map(drop),
                    (false, false) => batch
                        .push(Launch::new(
                            Ok(pipeline),
                            launch,
                            (&*a, &*b, &mut *out, n),
                            |(a, b, out, n)| {
                                Ok([
                                    Argument::read(a),
                                    Argument::read(b),
                                    Argument::write(out),
                                    Argument::value(*n)?,
                                ])
                            },
                        ))
                        .map(drop),
                }
            }
        })?;
        let start = Instant::now();
        let result = out.read().wait();
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

pub(super) struct Histogram {
    case: Case,
    pipeline: Pipeline,
    n: u32,
    values: Vec<u32>,
    launch: LaunchConfig<256>,
    input: Buffer<u32>,
    bins: Buffer<u32>,
}

impl Histogram {
    pub(super) fn new(device: &Device, config: &Config, variant: &str) -> Result<Self> {
        let n = config.elements;
        let mut case = Case::new("histogram", variant, vec![n, 16]);
        let pipeline = case
            .pipelines(device, config, "cooperation", &["histogram"])?
            .remove(0);
        let values = (0..n).map(|i| i.wrapping_mul(17) % 23).collect::<Vec<_>>();
        let start = Instant::now();
        let input = device.buffer_zeroed::<u32>(n as usize)?;
        let bins = device.buffer_zeroed::<u32>(16)?;
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipeline,
            n,
            values,
            launch: LaunchConfig::<256>::for_elements(n)?,
            input,
            bins,
        })
    }
}

impl Candidate for Histogram {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let (input, bins) = (&mut self.input, &mut self.bins);
        let start = Instant::now();
        let mut expected = [0_u32; 16];
        for &value in &self.values {
            expected[(value & 15) as usize] += 1;
        }
        let cpu = elapsed(start);
        let start = Instant::now();
        input.write().wait().copy_from_slice(&self.values);
        bins.write().wait().fill(0);
        let copy = elapsed(start);
        let start = Instant::now();
        input.upload();
        bins.upload();
        let upload = elapsed(start);
        let (pipeline, launch, n) = (&self.pipeline, self.launch, self.n);
        let mut sample = dispatch(|batch| {
            // SAFETY: input covers n, all writes are relaxed atomics into 16 initialized bins.
            unsafe {
                batch
                    .push(Launch::new(
                        Ok(pipeline),
                        launch,
                        (&*input, &mut *bins, n),
                        |(input, bins, n)| {
                            Ok([
                                Argument::read(input),
                                Argument::atomic(bins),
                                Argument::value(*n)?,
                            ])
                        },
                    ))
                    .map(drop)
            }
        })?;
        let start = Instant::now();
        let result = bins.read().wait();
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
