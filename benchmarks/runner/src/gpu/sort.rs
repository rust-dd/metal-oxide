use super::*;
use ::sort::{Step, bitonic::TILE, first_difference, padded_len, plan};
use metal_oxide::{Argument, Buffer, Dim3, LaunchConfig};

pub(super) struct Sort {
    case: Case,
    pipelines: Vec<Pipeline>,
    n: u32,
    values: Vec<u32>,
    steps: Vec<Step>,
    launch: LaunchConfig<256>,
    keys: Buffer<u32>,
}

impl Sort {
    pub(super) fn new(device: &Device, config: &Config, variant: &str) -> Result<Self> {
        let n = config.sort;
        let mut case = Case::new("sort", variant, vec![n]);
        let entries = ["sort_local", "merge_global", "merge_local"];
        let pipelines = case.pipelines(device, config, "sort", &entries)?;
        let values = (0..n)
            .map(|i| i.wrapping_mul(2_654_435_761) % (n / 2 + 1))
            .collect::<Vec<_>>();
        let len = padded_len(n);
        let start = Instant::now();
        let keys = device.buffer_zeroed::<u32>(len as usize)?;
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipelines,
            n,
            values,
            steps: plan(len),
            launch: LaunchConfig::<256>::new(Dim3::x(len / TILE)),
            keys,
        })
    }
}

impl Candidate for Sort {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let (n, keys) = (self.n as usize, &mut self.keys);
        let start = Instant::now();
        let mut expected = self.values.clone();
        expected.sort_unstable();
        let cpu = elapsed(start);
        let start = Instant::now();
        let (input, padding) = keys.write().wait().split_at_mut(n);
        input.copy_from_slice(&self.values);
        padding.fill(u32::MAX);
        let copy = elapsed(start);
        let start = Instant::now();
        keys.upload();
        let upload = elapsed(start);
        let (pipelines, launch) = (&self.pipelines, self.launch);
        let mut sample = dispatch(|batch| {
            for &step in &self.steps {
                // SAFETY: `steps` is `plan(keys.len())`, and `launch` is 1-D with a block per tile.
                unsafe { push(batch, pipelines, launch, keys, step)? };
            }
            Ok(())
        })?;
        let start = Instant::now();
        let result = keys.read().wait();
        sample.readback_ns = elapsed(start);
        let (sorted, padding) = result.split_at(n);
        let case = &mut self.case;
        if let Some(i) = first_difference(sorted, &expected) {
            panic!(
                "{} {}: key {i} is {} instead of {}",
                case.name, case.variant, sorted[i], expected[i]
            );
        }
        assert!(
            padding.iter().all(|&key| key == u32::MAX),
            "{} {}",
            case.name,
            case.variant
        );
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

/// Pushes `step` of `plan(keys.len())` for the kernels or their twin, which share parameters;
/// `launch` is one-dimensional with a block per tile of `keys`, which nothing else uses.
unsafe fn push(
    batch: &mut Batch,
    pipelines: &[Pipeline],
    launch: LaunchConfig<256>,
    keys: &mut Buffer<u32>,
    step: Step,
) -> metal_oxide::Result<()> {
    // SAFETY: on the caller's one-dimensional grid of a block per tile of at most 2^31 keys, plan
    // steps keep every stride a power of two dividing the thread count, as each kernel requires.
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
