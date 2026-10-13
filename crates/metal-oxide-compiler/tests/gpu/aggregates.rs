use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_uninitialized_private_arrays_execute() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/uninit_array.rs",
        "running_sums",
    )?;
    let input = device.buffer_from_slice(&(1..=64).map(|v| v as f32).collect::<Vec<_>>())?;
    for count in [0_u32, 1, 63, 64, 65] {
        let mut out = device.buffer_from_slice(&[-1.0_f32; 65])?;
        // SAFETY: one thread reads at most 64 inputs and writes at most 64 of 65 outputs.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut out, &input, count),
                |(out, input, count)| {
                    Ok([
                        Argument::write(out),
                        Argument::read(input),
                        Argument::value::<u32>(*count)?,
                    ])
                },
            )
        }
        .wait()?;
        let written = count.min(64) as usize;
        let expected = (0..65)
            .map(|i| {
                if i < written {
                    ((i + 1) * (i + 2) / 2) as f32
                } else {
                    -1.0
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(out.read().wait(), expected, "count={count}");
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_aggregate_copies_constants_and_generic_helpers_execute() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let samples = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/aggregates.rs",
        "aggregates",
    )?;
    let constants = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/aggregate_constants.rs",
        "aggregate_constants",
    )?;
    for seed in [-2.0_f32, 0.0, 10.0] {
        for (pipeline, expected) in [
            (
                &samples,
                vec![seed, seed + 1.0, seed + 3.0, seed + 1.0, seed],
            ),
            (&constants, vec![7.0, 3.0, seed, 4.0]),
        ] {
            let mut out = device.buffer_zeroed::<f32>(expected.len())?;
            // SAFETY: one thread writes the matching number of elements in a distinct output buffer.
            unsafe {
                Launch::new(
                    Ok(pipeline),
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    (&mut out, seed),
                    |(out, seed)| Ok([Argument::write(out), Argument::value::<f32>(*seed)?]),
                )
            }
            .wait()?;
            assert_eq!(out.read().wait(), expected);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rust_nested_aggregate_updates_across_control_flow_execute() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/aggregate_flow.rs",
        "aggregate_flow",
    )?;
    for (mode, expected) in [
        (0, [10.0, 11.0, 12.0, 13.0, 13.0]),
        (1, [12.0, 11.0, 12.0, 23.0, 13.0]),
        (2, [11.0, 11.0, 12.0, 23.0, 13.0]),
        (5, [7.0, 11.0, 12.0, 53.0, 13.0]),
    ] {
        let mut out = device.buffer_zeroed::<f32>(5)?;
        // SAFETY: one thread writes five initialized f32 elements; mode does not change the output bounds.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut out, 10.0, mode),
                |(out, v1, mode)| {
                    Ok([
                        Argument::write(out),
                        Argument::value::<f32>(*v1)?,
                        Argument::value::<u32>(*mode)?,
                    ])
                },
            )
        }
        .wait()?;
        assert_eq!(out.read().wait(), expected);
    }
    Ok(())
}
