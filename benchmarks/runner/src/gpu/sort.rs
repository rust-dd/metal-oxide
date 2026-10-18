use super::*;
use ::sort::{Step, bitonic::TILE, padded_len, plan};
use metal_oxide::{Argument, Buffer, Dim3, LaunchConfig};

pub(super) fn run(device: &Device, config: &Config, variant: &str) -> Result<Case> {
    let n = config.sort;
    let mut case = Case::new("sort", variant, vec![n]);
    let entries = ["sort_local", "merge_global", "merge_local"];
    let pipelines = case.pipelines(device, config, "sort", &entries)?;
    let values = (0..n)
        .map(|i| i.wrapping_mul(2_654_435_761) % (n / 2 + 1))
        .collect::<Vec<_>>();
    let len = padded_len(n);
    let start = Instant::now();
    let mut keys = device.buffer_zeroed::<u32>(len as usize)?;
    case.allocation_ns = elapsed(start);
    let steps = plan(len);
    let launch = LaunchConfig::<256>::new(Dim3::x(len / TILE));
    for iteration in 0..config.warmup + config.samples {
        let start = Instant::now();
        let mut expected = values.clone();
        expected.sort_unstable();
        let cpu = elapsed(start);
        let start = Instant::now();
        let (input, padding) = keys.write().wait().split_at_mut(n as usize);
        input.copy_from_slice(&values);
        padding.fill(u32::MAX);
        let copy = elapsed(start);
        let start = Instant::now();
        keys.upload();
        let upload = elapsed(start);
        let mut sample = dispatch(|batch| {
            for &step in &steps {
                // SAFETY: `steps` is `plan(keys.len())`, and `launch` is 1-D with a block per tile.
                unsafe { push(batch, &pipelines, launch, &mut keys, step)? };
            }
            Ok(())
        })?;
        let start = Instant::now();
        let result = keys.read().wait();
        sample.readback_ns = elapsed(start);
        let (sorted, padding) = result.split_at(n as usize);
        assert_eq!(sorted, expected, "sort {variant}");
        assert!(padding.iter().all(|&key| key == u32::MAX), "sort {variant}");
        sample.cpu_reference_ns = Some(cpu);
        sample.input_copy_ns = copy;
        sample.upload_ns = upload;
        if iteration >= config.warmup {
            case.samples.push(sample);
        }
    }
    case.finish()
}

/// Pushes `step` of `plan(keys.len())` for the kernels or their twin, which share parameters;
/// `launch` is one-dimensional with a block per tile of `keys`, which nothing else uses.
unsafe fn push(
    batch: &mut Batch,
    pipelines: &[Pipeline],
    launch: LaunchConfig<256>,
    keys: &mut Buffer<u32>,
    step: Step,
) -> metal_oxide::Result<()> {
    // SAFETY: on the caller's one-dimensional grid of a block per tile, plan steps keep every
    // stride a power of two that divides the thread count, as each kernel requires.
    unsafe {
        match step {
            Step::SortLocal => batch
                .push(Launch::new(Ok(&pipelines[0]), launch, (keys,), |(keys,)| {
                    Ok([Argument::read_write(keys)])
                }))
                .map(drop),
            Step::MergeGlobal { j, k } => batch
                .push(Launch::new(
                    Ok(&pipelines[1]),
                    launch,
                    (keys, j, k),
                    |(keys, j, k)| {
                        Ok([
                            Argument::read_write(keys),
                            Argument::value(*j)?,
                            Argument::value(*k)?,
                        ])
                    },
                ))
                .map(drop),
            Step::MergeLocal { k } => batch
                .push(Launch::new(
                    Ok(&pipelines[2]),
                    launch,
                    (keys, k),
                    |(keys, k)| Ok([Argument::read_write(keys), Argument::value(*k)?]),
                ))
                .map(drop),
        }
    }
}
