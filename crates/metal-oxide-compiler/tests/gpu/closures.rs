use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn closure_kernels_match_the_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/closures.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let n = 1000;
    let values = (0..n).map(|i| (i % 13) as f32).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let config = LaunchConfig::<256>::for_elements(n as u32)?;
    let pipeline = Pipeline::new(&device, &module, "closures")?;
    let mut out = device.buffer_zeroed::<f32>(n)?;
    // SAFETY: the grid covers every element; each thread writes only its own.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            config,
            (&input, &mut out, 2.5_f32),
            |(input, out, factor)| {
                Ok([
                    Argument::read(input),
                    Argument::write(out),
                    Argument::value(*factor)?,
                ])
            },
        )
        .wait()?;
    }
    let expected = values
        .iter()
        .map(|v| v * 2.5 + (v + 2.5))
        .collect::<Vec<_>>();
    assert_eq!(out.read().wait(), expected);
    let pipeline = Pipeline::new(&device, &module, "accumulate")?;
    let mut out = device.buffer_zeroed::<f32>(n)?;
    // SAFETY: the grid covers every element; each thread writes only its own.
    unsafe {
        Launch::new(Ok(&pipeline), config, (&input, &mut out), |(input, out)| {
            Ok([Argument::read(input), Argument::write(out)])
        })
        .wait()?;
    }
    let total = values.iter().sum::<f32>();
    let expected = (0..n)
        .map(|i| total + values[(i + 1) % n])
        .collect::<Vec<_>>();
    assert_eq!(out.read().wait(), expected);
    Ok(())
}
