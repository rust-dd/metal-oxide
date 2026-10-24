use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn guarded_casts_reduce_partial_blocks_and_preserve_padding() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/cast_bounds.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let reduce = Pipeline::new(&device, &module, "reduce_cast")?;
    let guarded = Pipeline::new(&device, &module, "guarded_cast")?;
    let narrowed = Pipeline::new(&device, &module, "narrowed_cast")?;
    let looped = Pipeline::new(&device, &module, "loop_cast")?;
    for count in [0_usize, 1, 63, 64, 65, 127, 129] {
        let values = (0..count).map(|i| (i as u32 + 1) * 3).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&values)?;
        let blocks = count.div_ceil(64).max(1);
        let mut output = device.buffer_from_slice(&vec![u32::MAX; blocks + 1])?;
        // SAFETY: 64 lanes share 64 slots; lane zero writes its block's output element.
        unsafe {
            Launch::new(
                Ok(&reduce),
                LaunchConfig::<64>::new(Dim3::x(blocks as u32)),
                (&input, &mut output),
                |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
            )
        }
        .wait()?;
        let expected = (0..blocks)
            .map(|block| {
                values
                    .iter()
                    .skip(block * 64)
                    .take(64)
                    .copied()
                    .sum::<u32>()
            })
            .chain([u32::MAX])
            .collect::<Vec<_>>();
        assert_eq!(output.read().wait(), expected, "count={count}");

        let mut output = device.buffer_from_slice(&[u32::MAX; 65])?;
        // SAFETY: each lane writes at most its own output slot; the final slot is padding.
        unsafe {
            Launch::new(
                Ok(&guarded),
                LaunchConfig::<64>::new(Dim3::x(1)),
                (&input, &mut output),
                |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
            )
        }
        .wait()?;
        let mut expected = [u32::MAX; 65];
        let copied = count.min(64);
        expected[..copied].copy_from_slice(&values[..copied]);
        assert_eq!(output.read().wait(), &expected, "count={count}");

        for index in [0_u64, 1, 63, 64, 255, 256, 257, u64::MAX] {
            let mut output = device.buffer_from_slice(&[u32::MAX])?;
            // SAFETY: one thread owns the single output element.
            unsafe {
                Launch::new(
                    Ok(&narrowed),
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    (&input, &mut output, index),
                    |(input, output, index)| {
                        Ok([
                            Argument::read(input),
                            Argument::write(output),
                            Argument::value(*index)?,
                        ])
                    },
                )
            }
            .wait()?;
            assert_eq!(
                output.read().wait(),
                [values
                    .get(index as u8 as usize)
                    .copied()
                    .unwrap_or(u32::MAX)]
            );
        }

        let mut output = device.buffer_zeroed::<u32>(1)?;
        // SAFETY: one thread owns the output and all input loads are bounds-checked.
        unsafe {
            Launch::new(
                Ok(&looped),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&input, &mut output, count as u32 + 3),
                |(input, output, count)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(output),
                        Argument::value(*count)?,
                    ])
                },
            )
        }
        .wait()?;
        assert_eq!(output.read().wait(), [values.iter().sum::<u32>()]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn changed_cast_inputs_still_report_bounds_failures() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    for configuration in [
        "reassign",
        "call_mutation",
        "join",
        "loop_mutation",
        "stale_guard",
        "narrowing",
    ] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/cast_bounds_bad.rs",
            &["--cfg", configuration, "-C", "overflow-checks=off"],
        );
        support::checked(output);
        let module = support::artifact::module(&device, &directory)?;
        let pipeline = Pipeline::new(&device, &module, "bad")?;
        let input = device.buffer_from_slice(&[11_u32, 22, 33, 44])?;
        let mut output = device.buffer_zeroed::<u32>(4)?;
        let other = if configuration == "narrowing" {
            256_u32
        } else {
            4
        };
        // SAFETY: every thread owns one output slot; invalid input indices must fail their checks.
        let failure = unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<4>::new(Dim3::x(1)),
                (&input, &mut output, other),
                |(input, output, other)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(output),
                        Argument::value(*other)?,
                    ])
                },
            )
        }
        .wait()
        .unwrap_err();
        let metal_oxide::Error::KernelFailed(failure) = failure else {
            panic!("{configuration}: {failure}");
        };
        assert_eq!(failure.message, "index out of bounds", "{configuration}");
    }
    Ok(())
}
