use super::*;
use metal_oxide::{Argument, LaunchConfig};

pub(super) fn run(device: &Device, config: &Config, variant: &str) -> Result<Vec<Case>> {
    Ok(vec![
        floats(device, config, variant, false)?,
        floats(device, config, variant, true)?,
        histogram(device, config, variant)?,
    ])
}

fn floats(device: &Device, config: &Config, variant: &str, reduction: bool) -> Result<Case> {
    let n = config.elements;
    let name = if reduction { "reduction" } else { "vec-add" };
    let mut case = Case::new(name, variant, vec![n]);
    let pipelines = case.pipelines(
        device,
        config,
        name,
        &[if reduction { "reduce" } else { "vec_add" }],
    )?;
    let values = (0..n).map(|i| (i % 31) as f32 * 0.25).collect::<Vec<_>>();
    let start = Instant::now();
    let mut a = device.buffer_zeroed::<f32>(n as usize)?;
    let mut b = device.buffer_zeroed::<f32>(if reduction { 0 } else { n } as usize)?;
    let mut out =
        device.buffer_zeroed::<f32>(if reduction { n.div_ceil(256) } else { n } as usize)?;
    case.allocation_ns = elapsed(start);
    let launch = LaunchConfig::<256>::for_elements(n)?;
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        let expected = if reduction {
            values
                .chunks(256)
                .map(|chunk| chunk.iter().sum::<f32>())
                .collect::<Vec<_>>()
        } else {
            values.iter().map(|x| x + x).collect()
        };
        let cpu = elapsed(start);
        let start = Instant::now();
        a.as_mut_slice().copy_from_slice(&values);
        if !reduction {
            b.as_mut_slice().copy_from_slice(&values);
        }
        let copy = elapsed(start);
        let start = Instant::now();
        a.upload();
        b.upload();
        out.upload();
        let upload = elapsed(start);
        // SAFETY: distinct buffers cover n inputs and the exact output shape; all 256 lanes participate.
        let mut sample = unsafe {
            dispatch(device, |batch| {
                if reduction {
                    batch.launch(
                        &pipelines[0],
                        launch,
                        &[
                            Argument::read(&a),
                            Argument::write(&mut out),
                            Argument::value(n)?,
                        ],
                    )
                } else {
                    batch.launch(
                        &pipelines[0],
                        launch,
                        &[
                            Argument::read(&a),
                            Argument::read(&b),
                            Argument::write(&mut out),
                            Argument::value(n)?,
                        ],
                    )
                }
            })?
        };
        let start = Instant::now();
        let result = out.as_slice();
        sample.readback_ns = elapsed(start);
        assert_eq!(result, expected, "{name} {variant}");
        sample.cpu_reference_ns = cpu;
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    case.finish()
}

fn histogram(device: &Device, config: &Config, variant: &str) -> Result<Case> {
    let n = config.elements;
    let mut case = Case::new("histogram", variant, vec![n, 16]);
    let pipelines = case.pipelines(device, config, "cooperation", &["histogram"])?;
    let values = (0..n).map(|i| i.wrapping_mul(17) % 23).collect::<Vec<_>>();
    let start = Instant::now();
    let mut input = device.buffer_zeroed::<u32>(n as usize)?;
    let mut bins = device.buffer_zeroed::<u32>(16)?;
    case.allocation_ns = elapsed(start);
    let launch = LaunchConfig::<256>::for_elements(n)?;
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        let mut expected = [0_u32; 16];
        for &value in &values {
            expected[(value & 15) as usize] += 1;
        }
        let cpu = elapsed(start);
        let start = Instant::now();
        input.as_mut_slice().copy_from_slice(&values);
        bins.as_mut_slice().fill(0);
        let copy = elapsed(start);
        let start = Instant::now();
        input.upload();
        bins.upload();
        let upload = elapsed(start);
        // SAFETY: input covers n, all writes are relaxed atomics into 16 initialized bins.
        let mut sample = unsafe {
            dispatch(device, |batch| {
                batch.launch(
                    &pipelines[0],
                    launch,
                    &[
                        Argument::read(&input),
                        Argument::atomic(&mut bins),
                        Argument::value(n)?,
                    ],
                )
            })?
        };
        let start = Instant::now();
        let result = bins.as_slice();
        sample.readback_ns = elapsed(start);
        assert_eq!(result, expected, "histogram {variant}");
        sample.cpu_reference_ns = cpu;
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    case.finish()
}
