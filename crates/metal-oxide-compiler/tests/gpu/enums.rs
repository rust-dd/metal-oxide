use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn enum_kernels_match_the_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/enums.rs", &[]);
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let n = 1000;
    let values = (0..n).map(|i| ((i * 37) % 101) as f32).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let area = |i: usize| {
        let size = values[i];
        match i % 4 {
            0 => 3.0 * size * size,
            1 => size * size,
            2 => size * (size + 1.0),
            _ => 0.0,
        }
    };
    let expected = [
        (0..n).map(area).collect::<Vec<_>>(),
        (0..n)
            .map(|i| values[..=i].iter().copied().fold(f32::MIN, f32::max))
            .collect(),
    ];
    for (kernel, expected) in ["areas", "largest_prefix"].into_iter().zip(expected) {
        let pipeline = Pipeline::new(&device, &module, kernel)?;
        let mut out = device.buffer_zeroed::<f32>(n)?;
        // SAFETY: the grid covers every element; each thread writes only its own.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<256>::for_elements(n as u32)?,
                (&input, &mut out),
                |(input, out)| Ok([Argument::read(input), Argument::write(out)]),
            )
            .wait()?;
        }
        assert_eq!(out.read().wait(), expected, "{kernel}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_failed_unwrap_reports_its_message_and_call() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/enum_methods.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let pipeline = Pipeline::new(&device, &module, "past_the_end")?;
    let values = device.buffer_from_slice(&[1_u32, 2, 3])?;
    let mut out = device.buffer_zeroed::<u32>(3)?;
    // SAFETY: the grid covers every element; each thread writes only its own.
    let result = unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<32>::for_elements(3)?,
            (&values, &mut out),
            |(values, out)| Ok([Argument::read(values), Argument::write(out)]),
        )
        .wait()
        .map(drop)
    };
    let Err(metal_oxide::Error::KernelFailed(failure)) = result else {
        panic!("expected a kernel failure, got {result:?}");
    };
    assert_eq!(
        (failure.message.as_str(), failure.line, failure.column),
        ("called `Option::unwrap()` on a `None` value", 40, 50)
    );
    Ok(())
}
