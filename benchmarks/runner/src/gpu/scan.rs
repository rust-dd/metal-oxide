use super::*;
use metal_oxide::{Argument, Buffer, LaunchConfig};

struct Level {
    local: Buffer<u32>,
    output: Buffer<u32>,
    totals: Buffer<u32>,
    n: u32,
}

pub(super) struct Scan {
    case: Case,
    pipelines: Vec<Pipeline>,
    values: Vec<u32>,
    input: Buffer<u32>,
    zero: Buffer<u32>,
    levels: Vec<Level>,
}

impl Scan {
    pub(super) fn new(device: &Device, config: &Config, variant: &str) -> Result<Self> {
        let n = config.elements;
        let mut case = Case::new("scan", variant, vec![n]);
        let pipelines = case.pipelines(
            device,
            config,
            "cooperation",
            &["block_scan", "add_offsets"],
        )?;
        let values = (0..n).map(|i| i % 7).collect::<Vec<_>>();
        let start = Instant::now();
        let input = device.buffer_zeroed::<u32>(n as usize)?;
        let zero = device.buffer_zeroed::<u32>(1)?;
        let mut levels = Vec::new();
        let mut size = n;
        loop {
            let blocks = size.div_ceil(256);
            levels.push(Level {
                local: device.buffer_zeroed(size as usize)?,
                output: device.buffer_zeroed(size as usize)?,
                totals: device.buffer_zeroed(blocks as usize)?,
                n: size,
            });
            if blocks == 1 {
                break;
            }
            size = blocks;
        }
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipelines,
            values,
            input,
            zero,
            levels,
        })
    }
}

impl Candidate for Scan {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let (input, zero, levels) = (&mut self.input, &mut self.zero, &mut self.levels);
        let start = Instant::now();
        let mut sum = 0_u32;
        let expected = self
            .values
            .iter()
            .map(|&v| {
                let before = sum;
                sum = sum.wrapping_add(v);
                before
            })
            .collect::<Vec<_>>();
        let cpu = elapsed(start);
        let start = Instant::now();
        input.write().wait().copy_from_slice(&self.values);
        let copy = elapsed(start);
        let start = Instant::now();
        input.upload();
        zero.upload();
        for level in levels.iter_mut() {
            level.local.upload();
            level.output.upload();
            level.totals.upload();
        }
        let upload = elapsed(start);
        let pipelines = &self.pipelines;
        let mut sample = dispatch(|batch| {
            // SAFETY: each level owns separate buffers covering its padded blocks; ordered passes
            // consume completed writes.
            unsafe {
                for index in 0..levels.len() {
                    let (previous, current) = levels.split_at_mut(index);
                    let source = if index == 0 {
                        &*input
                    } else {
                        &previous[index - 1].totals
                    };
                    let level = &mut current[0];
                    batch.push(Launch::new(
                        Ok(&pipelines[0]),
                        LaunchConfig::<256>::for_elements(level.n)?,
                        (source, &mut level.local, &mut level.totals, level.n),
                        |(source, prefix, totals, n)| {
                            Ok([
                                Argument::read(source),
                                Argument::write(prefix),
                                Argument::write(totals),
                                Argument::value(*n)?,
                            ])
                        },
                    ))?;
                }
                for index in (0..levels.len()).rev() {
                    let (children, parents) = levels.split_at_mut(index + 1);
                    let source = parents.first().map_or(&*zero, |parent| &parent.output);
                    let level = &mut children[index];
                    batch.push(Launch::new(
                        Ok(&pipelines[1]),
                        LaunchConfig::<256>::for_elements(level.n)?,
                        (&level.local, source, &mut level.output, level.n),
                        |(local, source, output, n)| {
                            Ok([
                                Argument::read(local),
                                Argument::read(source),
                                Argument::write(output),
                                Argument::value(*n)?,
                            ])
                        },
                    ))?;
                }
                Ok(())
            }
        })?;
        let start = Instant::now();
        let result = levels[0].output.read().wait();
        let total = levels.last().unwrap().totals.read().wait()[0];
        sample.readback_ns = elapsed(start);
        let case = &mut self.case;
        assert_eq!(result, expected, "{} {}", case.name, case.variant);
        assert_eq!(total, sum);
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
