use fft::{cpu, half_spans, launch, stockham};
use metal_oxide::{
    Argument, Batch, Buffer, Device, Launch, LaunchConfig, Module, Operation, Pipeline, Result,
    sequence,
};

/// The real and imaginary parts of the same rows.
type Pair = (Buffer<f32>, Buffer<f32>);

/// The generated kernel or its handwritten MSL twin.
pub(super) enum Program {
    Generated(Device),
    Twin(Pipeline),
}

impl Program {
    pub(super) fn twin(device: &Device) -> Result<Self> {
        let source = include_str!("../../../benchmarks/reference-msl/fft.metal");
        let module = Module::from_source(device, source)?;
        Ok(Self::Twin(Pipeline::new(device, &module, "fft_stage")?))
    }

    /// Pushes the stage of `half_span`, a power of two below `n`, from `input` to `output`, pairs
    /// that hold `n` values, a power of two, per row of the two-dimensional `launch`.
    unsafe fn push(
        &self,
        batch: &mut Batch,
        launch: LaunchConfig<256>,
        input: &Pair,
        output: &mut Pair,
        n: u32,
        half_span: u32,
    ) -> Result<()> {
        // SAFETY: the caller's rows, `n` and `half_span` meet the kernel's contract.
        unsafe {
            match self {
                Self::Generated(device) => batch
                    .push(stockham::fft_stage(
                        device,
                        launch,
                        &input.0,
                        &input.1,
                        &mut output.0,
                        &mut output.1,
                        n,
                        half_span,
                    ))
                    .map(drop),
                Self::Twin(pipeline) => batch
                    .push(Launch::new(
                        Ok(pipeline),
                        launch,
                        (
                            &input.0,
                            &input.1,
                            &mut output.0,
                            &mut output.1,
                            n,
                            half_span,
                        ),
                        |(input_re, input_im, output_re, output_im, n, half_span)| {
                            Ok([
                                Argument::read(input_re),
                                Argument::read(input_im),
                                Argument::write(output_re),
                                Argument::write(output_im),
                                Argument::value(*n)?,
                                Argument::value(*half_span)?,
                            ])
                        },
                    ))
                    .map(drop),
            }
        }
    }
}

/// The transform of every row of `n` values in `re` and `im` by `program`, which runs all
/// stages in one command buffer, from a pair of buffers to the other and back.
pub(super) fn fft(
    device: &Device,
    program: &Program,
    n: u32,
    re: &[f32],
    im: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let rows = u32::try_from(re.len() / n as usize).expect("at most u32::MAX rows");
    let unwritten = vec![f32::NAN; re.len()];
    let mut pairs = [
        (device.buffer_from_slice(re)?, device.buffer_from_slice(im)?),
        (
            device.buffer_from_slice(&unwritten)?,
            device.buffer_from_slice(&unwritten)?,
        ),
    ];
    let launch = launch(n, rows);
    sequence(|batch| {
        let [a, b] = &mut pairs;
        for (stage, half_span) in half_spans(n).enumerate() {
            let (input, output) = if stage % 2 == 0 {
                (&*a, &mut *b)
            } else {
                (&*b, &mut *a)
            };
            // SAFETY: both pairs hold `n` values per row of `launch`, and `half_spans` yields
            // powers of two below `n`, itself a power of two.
            unsafe { program.push(batch, launch, input, output, n, half_span)? };
        }
        Ok(())
    })
    .wait()?;
    let (re, im) = &pairs[half_spans(n).count() % 2];
    Ok((re.read().wait().to_vec(), im.read().wait().to_vec()))
}

/// `len` complex values whose parts lie in [-1, 1).
pub(super) fn signal(len: u32) -> (Vec<f32>, Vec<f32>) {
    let part = |i: u32| (i.wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0;
    let re = (0..len).map(part).collect();
    let im = (0..len).map(|i| part(i ^ 0x5bd1_e995)).collect();
    (re, im)
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

pub(super) fn verify() -> Result<()> {
    let device = Device::system_default()?;
    let generated = Program::Generated(device.clone());
    let twin = Program::twin(&device)?;
    println!("Device: {}", device.name());
    for n in [2, 8, 1024, 4096] {
        for rows in [1, 3] {
            let (re, im) = signal(n * rows);
            let expected = cpu::reference(n as usize, &re, &im);
            let rust = fft(&device, &generated, n, &re, &im)?;
            let msl = fft(&device, &twin, n, &re, &im)?;
            let error = cpu::error(n as usize, &expected, &rust.0, &rust.1);
            assert!(
                error <= 1.0,
                "Rust, n={n}, rows={rows}: {error} tolerances off"
            );
            assert!(
                bits(&rust.0) == bits(&msl.0) && bits(&rust.1) == bits(&msl.1),
                "Rust and MSL differ bitwise, n={n}, rows={rows}"
            );
            println!(
                "fft n={n} rows={rows}: Rust and MSL agree bitwise, {error:.3} of the CPU tolerance"
            );
        }
    }
    Ok(())
}
