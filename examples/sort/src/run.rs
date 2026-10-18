use metal_oxide::{
    Argument, Batch, Buffer, Device, Dim3, Launch, LaunchConfig, Module, Operation, Pipeline,
    Result, sequence,
};
use sort::{Step, bitonic, padded_len, plan};

/// The generated kernels or their handwritten MSL twin.
pub(super) enum Program {
    Generated(Device),
    Twin([Pipeline; 3]),
}

impl Program {
    pub(super) fn twin(device: &Device) -> Result<Self> {
        let source = include_str!("../../../benchmarks/reference-msl/sort.metal");
        let module = Module::from_source(device, source)?;
        let pipeline = |entry| Pipeline::new(device, &module, entry);
        let pipelines = [
            pipeline("sort_local")?,
            pipeline("merge_global")?,
            pipeline("merge_local")?,
        ];
        Ok(Self::Twin(pipelines))
    }

    /// Pushes `step`, a step of `plan(keys.len())`, over every tile of `keys`, which nothing else
    /// uses meanwhile.
    unsafe fn push(&self, batch: &mut Batch, keys: &mut Buffer<u32>, step: Step) -> Result<()> {
        let launch = LaunchConfig::<256>::new(Dim3::x(keys.len() as u32 / bitonic::TILE));
        // SAFETY: a one-dimensional grid of a block per tile, and steps of `plan(keys.len())` with
        // strides that are powers of two dividing the thread count, meet each kernel's contract.
        unsafe {
            match (self, step) {
                (Self::Generated(device), Step::SortLocal) => batch
                    .push(bitonic::sort_local(device, launch, keys))
                    .map(drop),
                (Self::Generated(device), Step::MergeGlobal { j, k }) => batch
                    .push(bitonic::merge_global(device, launch, keys, j, k))
                    .map(drop),
                (Self::Generated(device), Step::MergeLocal { k }) => batch
                    .push(bitonic::merge_local(device, launch, keys, k))
                    .map(drop),
                (Self::Twin(pipelines), Step::SortLocal) => batch
                    .push(Launch::new(Ok(&pipelines[0]), launch, (keys,), |(keys,)| {
                        Ok([Argument::read_write(keys)])
                    }))
                    .map(drop),
                (Self::Twin(pipelines), Step::MergeGlobal { j, k }) => batch
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
                (Self::Twin(pipelines), Step::MergeLocal { k }) => batch
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
}

/// `keys` sorted by `program`, which runs the whole plan over the padded keys in one command
/// buffer; the padding must come out last.
pub(super) fn sort(device: &Device, program: &Program, keys: &[u32]) -> Result<Vec<u32>> {
    let len = padded_len(u32::try_from(keys.len()).expect("at most 2^31 keys"));
    let mut padded = keys.to_vec();
    padded.resize(len as usize, u32::MAX);
    let mut buffer = device.buffer_from_slice(&padded)?;
    let steps = plan(len);
    sequence(|batch| {
        for &step in &steps {
            // SAFETY: the steps come from `plan(buffer.len())`; only this batch uses the buffer.
            unsafe { program.push(batch, &mut buffer, step)? };
        }
        Ok(())
    })
    .wait()?;
    let (sorted, padding) = buffer.read().wait().split_at(keys.len());
    assert!(
        padding.iter().all(|&key| key == u32::MAX),
        "the padding sorts last"
    );
    Ok(sorted.to_vec())
}

/// `n` keys with duplicates, among them 0 and `u32::MAX`, the padding's value.
pub(super) fn keys(n: u32) -> Vec<u32> {
    (0..n)
        .map(|i| match i % 16 {
            0 => u32::MAX,
            1 => 0,
            _ => i.wrapping_mul(2_654_435_761) % (n / 4 + 1),
        })
        .collect()
}

pub(super) fn verify() -> Result<()> {
    let device = Device::system_default()?;
    let programs = [
        ("Rust", Program::Generated(device.clone())),
        ("MSL", Program::twin(&device)?),
    ];
    println!("Device: {}", device.name());
    for n in [0, 1, 2, 255, 512, 513, 4096, 100_003] {
        let input = keys(n);
        let mut expected = input.clone();
        expected.sort_unstable();
        for (name, program) in &programs {
            assert_eq!(sort(&device, program, &input)?, expected, "{name}, n={n}");
        }
        println!("sort n={n}: Rust, MSL and CPU agree");
    }
    Ok(())
}
