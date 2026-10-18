use super::*;
use ::fft::{cpu, half_spans, launch};
use metal_oxide::{Argument, Buffer, LaunchConfig};
use std::cell::OnceCell;

/// The first result of a case's candidates, which every sample of both must match bitwise.
pub(super) type First = OnceCell<(Vec<f32>, Vec<f32>)>;

/// The real and imaginary parts of the same rows.
type Pair = (Buffer<f32>, Buffer<f32>);

pub(super) struct Fft<'a> {
    case: Case,
    pipeline: Pipeline,
    n: u32,
    values: (Vec<f32>, Vec<f32>),
    launch: LaunchConfig<256>,
    pairs: [Pair; 2],
    first: &'a First,
}

impl<'a> Fft<'a> {
    pub(super) fn new(
        device: &Device,
        config: &Config,
        variant: &str,
        first: &'a First,
    ) -> Result<Self> {
        let [n, rows] = config.fft;
        let mut case = Case::new("fft", variant, config.fft.to_vec());
        let pipeline = case
            .pipelines(device, config, "fft", &["fft_stage"])?
            .remove(0);
        let len = n * rows;
        let part = |i: u32| (i.wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0;
        let values = (
            (0..len).map(part).collect(),
            (0..len).map(|i| part(i ^ 0x5bd1_e995)).collect(),
        );
        let start = Instant::now();
        let pair = || -> metal_oxide::Result<Pair> {
            let len = len as usize;
            Ok((device.buffer_zeroed(len)?, device.buffer_zeroed(len)?))
        };
        let pairs = [pair()?, pair()?];
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipeline,
            n,
            values,
            launch: launch(n, rows),
            pairs,
            first,
        })
    }
}

impl Candidate for Fft<'_> {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let (n, (re, im)) = (self.n, &self.values);
        let start = Instant::now();
        let expected = cpu::reference(n as usize, re, im);
        let cpu = elapsed(start);
        let [a, b] = &mut self.pairs;
        let start = Instant::now();
        a.0.write().wait().copy_from_slice(re);
        a.1.write().wait().copy_from_slice(im);
        let copy = elapsed(start);
        let start = Instant::now();
        a.0.upload();
        a.1.upload();
        let upload = elapsed(start);
        let (pipeline, launch) = (&self.pipeline, self.launch);
        let mut sample = dispatch(|batch| {
            for (stage, half_span) in half_spans(n).enumerate() {
                let (input, output) = if stage % 2 == 0 {
                    (&*a, &mut *b)
                } else {
                    (&*b, &mut *a)
                };
                // SAFETY: both pairs hold `n` values per row of `launch`, and `half_spans` yields
                // powers of two below `n`, itself a power of two.
                unsafe { push(batch, pipeline, launch, input, output, n, half_span)? };
            }
            Ok(())
        })?;
        let result = &self.pairs[half_spans(n).count() % 2];
        let start = Instant::now();
        let (result_re, result_im) = (result.0.read().wait(), result.1.read().wait());
        sample.readback_ns = elapsed(start);
        let case = &mut self.case;
        let error = cpu::error(n as usize, &expected, result_re, result_im);
        assert!(
            error <= 1.0,
            "{} {}: {error} tolerances off",
            case.name,
            case.variant
        );
        let first = self
            .first
            .get_or_init(|| (result_re.to_vec(), result_im.to_vec()));
        let same = |a: &[f32], b: &[f32]| {
            a.iter()
                .map(|x| x.to_bits())
                .eq(b.iter().map(|x| x.to_bits()))
        };
        assert!(
            same(&first.0, result_re) && same(&first.1, result_im),
            "{} {}: differs bitwise from the first result",
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

/// Pushes the kernel's or the twin's stage of `half_span`, a power of two below `n`, from `input`
/// to `output`, pairs of `n` values, a power of two, per row of the two-dimensional `launch`.
unsafe fn push(
    batch: &mut Batch,
    pipeline: &Pipeline,
    launch: LaunchConfig<256>,
    input: &Pair,
    output: &mut Pair,
    n: u32,
    half_span: u32,
) -> metal_oxide::Result<()> {
    // SAFETY: the caller's rows, `n` and `half_span` meet the kernel's contract.
    unsafe {
        batch
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
            .map(drop)
    }
}
