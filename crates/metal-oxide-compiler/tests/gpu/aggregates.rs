use super::*;

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
                device.launch(
                    pipeline,
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    &[Argument::write(&mut out), Argument::value::<f32>(seed)?],
                )?;
            }
            assert_eq!(out.as_slice(), expected);
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
            device.launch(
                &pipeline,
                LaunchConfig::<1>::new(Dim3::x(1)),
                &[
                    Argument::write(&mut out),
                    Argument::value::<f32>(10.0)?,
                    Argument::value::<u32>(mode)?,
                ],
            )?;
        }
        assert_eq!(out.as_slice(), expected);
    }
    Ok(())
}
