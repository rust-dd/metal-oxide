use metal_oxide::{
    Argument, Batch, Buffer, Device, Launch, LaunchConfig, Module, Operation, Pipeline, Result,
    sequence,
};
use stencil::{five_point, launch, reference, values};

/// The generated kernel or its handwritten MSL twin.
pub(super) enum Program {
    Generated(Device),
    Twin(Pipeline),
}

impl Program {
    pub(super) fn twin(device: &Device) -> Result<Self> {
        let source = include_str!("../../../benchmarks/reference-msl/stencil.metal");
        let module = Module::from_source(device, source)?;
        Ok(Self::Twin(Pipeline::new(device, &module, "jacobi")?))
    }

    /// Pushes a sweep from `input` into `output`, grids of `width * height` values, at most
    /// `u32::MAX`, on the two-dimensional `launch`.
    unsafe fn push(
        &self,
        batch: &mut Batch,
        launch: LaunchConfig<16, 16>,
        input: &Buffer<f32>,
        output: &mut Buffer<f32>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        match self {
            Self::Generated(device) => batch
                .push(five_point::jacobi(
                    device, launch, input, output, width, height,
                ))
                .map(drop),
            // SAFETY: the caller's grids and launch keep the twin in bounds with a cell per
            // thread; the borrows exclude other users, and the batch orders the other sweeps.
            Self::Twin(pipeline) => unsafe {
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
            },
        }
    }
}

/// `values`, a grid of `width x height` values, after `iterations` sweeps by `program`, which
/// runs them all in one command buffer, from one buffer to the other and back.
pub(super) fn jacobi(
    device: &Device,
    program: &Program,
    width: u32,
    height: u32,
    iterations: u32,
    values: &[f32],
) -> Result<Vec<f32>> {
    let cells = u64::from(width) * u64::from(height);
    assert!(
        values.len() as u64 == cells && cells <= u64::from(u32::MAX),
        "a grid of {width} x {height} values, at most u32::MAX"
    );
    let unwritten = vec![f32::NAN; values.len()];
    let mut grids = [
        device.buffer_from_slice(values)?,
        device.buffer_from_slice(&unwritten)?,
    ];
    let launch = launch(width, height);
    sequence(|batch| {
        let [a, b] = &mut grids;
        for sweep in 0..iterations {
            let (input, output) = if sweep % 2 == 0 {
                (&*a, &mut *b)
            } else {
                (&*b, &mut *a)
            };
            // SAFETY: both grids hold the asserted `width * height` values, and `launch` is the
            // two-dimensional one of the grid.
            unsafe { program.push(batch, launch, input, output, width, height)? };
        }
        Ok(())
    })
    .wait()?;
    Ok(grids[iterations as usize % 2].read().wait().to_vec())
}

/// The first cell where `a` and `b` differ bitwise.
fn difference(a: &[f32], b: &[f32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(a, b)| a.to_bits() != b.to_bits())
}

pub(super) fn verify() -> Result<()> {
    let device = Device::system_default()?;
    let programs = [
        ("Rust", Program::Generated(device.clone())),
        ("MSL", Program::twin(&device)?),
    ];
    println!("Device: {}", device.name());
    for (width, height) in [(1, 1), (2, 3), (17, 33), (256, 256)] {
        let input = values(width * height);
        for iterations in [1, 3] {
            let expected = reference(&input, width as usize, height as usize, iterations);
            for (name, program) in &programs {
                let result = jacobi(&device, program, width, height, iterations, &input)?;
                let differs = difference(&result, &expected);
                assert!(
                    differs.is_none(),
                    "{name}, {width}x{height} iterations={iterations}: cell {differs:?} differs"
                );
            }
            println!(
                "stencil {width}x{height} iterations={iterations}: Rust, MSL and CPU agree bitwise"
            );
        }
    }
    Ok(())
}
