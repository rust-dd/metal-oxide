use super::*;
use ::stencil::{launch, reference, values};
use metal_oxide::{Argument, Buffer, LaunchConfig};

pub(super) struct Stencil {
    case: Case,
    pipeline: Pipeline,
    /// Width, height and iterations.
    shape: [u32; 3],
    values: Vec<f32>,
    launch: LaunchConfig<16, 16>,
    grids: [Buffer<f32>; 2],
}

impl Stencil {
    pub(super) fn new(device: &Device, config: &Config, variant: &str) -> Result<Self> {
        let [width, height, _] = config.stencil;
        let mut case = Case::new("stencil", variant, config.stencil.to_vec());
        let pipeline = case
            .pipelines(device, config, "stencil", &["jacobi"])?
            .remove(0);
        let cells = width * height;
        let values = values(cells);
        let start = Instant::now();
        let grids = [
            device.buffer_zeroed(cells as usize)?,
            device.buffer_zeroed(cells as usize)?,
        ];
        case.allocation_ns = elapsed(start);
        Ok(Self {
            case,
            pipeline,
            shape: config.stencil,
            values,
            launch: launch(width, height),
            grids,
        })
    }
}

impl Candidate for Stencil {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let [width, height, iterations] = self.shape;
        let start = Instant::now();
        let expected = reference(&self.values, width as usize, height as usize, iterations);
        let cpu = elapsed(start);
        let [a, b] = &mut self.grids;
        let start = Instant::now();
        a.write().wait().copy_from_slice(&self.values);
        let copy = elapsed(start);
        let start = Instant::now();
        a.upload();
        let upload = elapsed(start);
        let (pipeline, launch) = (&self.pipeline, self.launch);
        let mut sample = dispatch(|batch| {
            for sweep in 0..iterations {
                let (input, output) = if sweep % 2 == 0 {
                    (&*a, &mut *b)
                } else {
                    (&*b, &mut *a)
                };
                // SAFETY: both grids hold the `width * height` values of the validated config,
                // below u32::MAX, and `launch` is the grid's two-dimensional one.
                unsafe { push(batch, pipeline, launch, input, output, width, height)? };
            }
            Ok(())
        })?;
        let start = Instant::now();
        let result = self.grids[iterations as usize % 2].read().wait();
        sample.readback_ns = elapsed(start);
        let case = &mut self.case;
        let same = result
            .iter()
            .map(|value| value.to_bits())
            .eq(expected.iter().map(|value| value.to_bits()));
        assert!(
            same,
            "{} {}: differs bitwise from the CPU reference",
            case.name, case.variant
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

/// Pushes the kernel's or the twin's sweep from `input` into `output`, grids of `width * height`
/// values, at most u32::MAX, on the two-dimensional `launch`.
unsafe fn push(
    batch: &mut Batch,
    pipeline: &Pipeline,
    launch: LaunchConfig<16, 16>,
    input: &Buffer<f32>,
    output: &mut Buffer<f32>,
    width: u32,
    height: u32,
) -> metal_oxide::Result<()> {
    // SAFETY: the caller's grids and launch keep both programs in bounds with a cell per thread;
    // the borrows exclude other users, and the batch orders the other sweeps around this one.
    unsafe {
        batch
            .push(Launch::new(
                Ok(pipeline),
                launch,
                (input, output, width, height),
                |(input, output, width, height)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(output),
                        Argument::value(*width)?,
                        Argument::value(*height)?,
                    ])
                },
            ))
            .map(drop)
    }
}
