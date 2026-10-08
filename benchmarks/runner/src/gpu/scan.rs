use super::*;
use metal_oxide::{Argument, Buffer, LaunchConfig};

struct Level {
    local: Buffer<u32>,
    output: Buffer<u32>,
    totals: Buffer<u32>,
    n: u32,
}

pub(super) fn run(device: &Device, config: &Config, variant: &str) -> Result<Case> {
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
    let mut input = device.buffer_zeroed::<u32>(n as usize)?;
    let mut zero = device.buffer_zeroed::<u32>(1)?;
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
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        let mut sum = 0_u32;
        let expected = values
            .iter()
            .map(|&v| {
                let before = sum;
                sum = sum.wrapping_add(v);
                before
            })
            .collect::<Vec<_>>();
        let cpu = elapsed(start);
        let start = Instant::now();
        input.as_mut_slice().copy_from_slice(&values);
        let copy = elapsed(start);
        let start = Instant::now();
        input.upload();
        zero.upload();
        for level in &mut levels {
            level.local.upload();
            level.output.upload();
            level.totals.upload();
        }
        let upload = elapsed(start);
        // SAFETY: each level owns separate buffers covering its padded blocks; ordered passes consume completed writes.
        let mut sample = unsafe {
            dispatch(device, |batch| {
                for index in 0..levels.len() {
                    let (previous, current) = levels.split_at_mut(index);
                    let source = if index == 0 {
                        &input
                    } else {
                        &previous[index - 1].totals
                    };
                    let level = &mut current[0];
                    batch.launch(
                        &pipelines[0],
                        LaunchConfig::<256>::for_elements(level.n)?,
                        &[
                            Argument::read(source),
                            Argument::write(&mut level.local),
                            Argument::write(&mut level.totals),
                            Argument::value(level.n)?,
                        ],
                    )?;
                }
                for index in (0..levels.len()).rev() {
                    let (children, parents) = levels.split_at_mut(index + 1);
                    let source = parents.first().map_or(&zero, |parent| &parent.output);
                    let level = &mut children[index];
                    batch.launch(
                        &pipelines[1],
                        LaunchConfig::<256>::for_elements(level.n)?,
                        &[
                            Argument::read(&level.local),
                            Argument::read(source),
                            Argument::write(&mut level.output),
                            Argument::value(level.n)?,
                        ],
                    )?;
                }
                Ok(())
            })?
        };
        let start = Instant::now();
        let result = levels[0].output.as_slice();
        let total = levels.last().unwrap().totals.as_slice()[0];
        sample.readback_ns = elapsed(start);
        assert_eq!(result, expected, "scan {variant}");
        assert_eq!(total, sum);
        sample.cpu_reference_ns = cpu;
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    case.finish()
}
