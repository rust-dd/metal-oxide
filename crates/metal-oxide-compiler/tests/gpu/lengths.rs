use super::*;

/// Launches `lengths` over `a` and `b` into a zeroed buffer of `out` elements.
fn lengths(
    batch: &mut metal_oxide::Batch,
    pipeline: &Pipeline,
    a: &metal_oxide::Buffer<f32>,
    b: &metal_oxide::Buffer<f32>,
    out: &mut metal_oxide::Buffer<f32>,
) -> metal_oxide::Result<()> {
    // SAFETY: each thread writes its own element of `out` within the bound lengths.
    unsafe {
        batch.push(Launch::new(
            Ok(pipeline),
            LaunchConfig::<4>::new(Dim3::x(2)),
            (a, b, out),
            |(a, b, out)| Ok([Argument::read(a), Argument::read(b), Argument::write(out)]),
        ))?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn every_dispatch_reads_the_lengths_of_its_own_buffers() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/buffer_lengths.rs",
        &[],
    );
    support::checked(output);
    let pipeline = Pipeline::new(
        &device,
        &support::artifact::module(&device, &directory)?,
        "lengths",
    )?;
    let short = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0])?;
    let long = device.buffer_from_slice(&[10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0])?;
    let mut four = device.buffer_zeroed::<f32>(4)?;
    let mut eight = device.buffer_zeroed::<f32>(8)?;
    sequence(|batch| {
        lengths(batch, &pipeline, &short, &long, &mut four)?;
        lengths(batch, &pipeline, &long, &short, &mut eight)
    })
    .wait()?;
    assert_eq!(four.read().wait(), [11.0, 22.0, 33.0, 6.0]);
    assert_eq!(
        eight.read().wait(),
        [11.0, 22.0, 33.0, 0.0, 0.0, 0.0, 0.0, 210.0]
    );
    Ok(())
}
