use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn generic_entrypoints_execute_concrete_types_constants_and_aliases() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/generic_entries.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    verify(
        &device,
        &module,
        "scale_f32",
        [1.5_f32, -2.0, 0.0],
        [6.0, -8.0, 0.0],
    )?;
    verify(
        &device,
        &module,
        "alias_f32",
        [1.5_f32, -2.0, 0.0],
        [6.0, -8.0, 0.0],
    )?;
    verify(&device, &module, "scale_i32", [-3_i32, 4, 0], [-24, 32, 0])?;
    verify(
        &device,
        &module,
        "scale_u32",
        [7_u32, 0, u32::MAX],
        [14, 0, u32::MAX - 1],
    )?;
    Ok(())
}

fn verify<T: metal_oxide::GpuValue + PartialEq + std::fmt::Debug>(
    device: &Device,
    module: &Module,
    name: &str,
    input: [T; 3],
    expected: [T; 3],
) -> metal_oxide::Result<()> {
    let pipeline = Pipeline::new(device, module, name)?;
    let mut output = device.buffer_from_slice(&input)?;
    let input = device.buffer_from_slice(&input)?;
    for block in [32, 16] {
        let config = DynamicLaunchConfig::for_elements(3, block)?;
        // SAFETY: the safe kernel bounds each write by its thread index and checks input length.
        let result = unsafe {
            Launch::new(
                Ok(&pipeline),
                config,
                (&input, &mut output),
                |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
            )
        }
        .wait();
        if block == 32 {
            result?;
            assert_eq!(output.read().wait(), expected, "{name}");
        } else {
            assert!(result.is_err(), "{name} accepted the wrong block shape");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn const_instances_keep_required_checks_and_report_the_instance_name() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/generic_checks.rs",
        &[],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )?;
    assert!(
        !abi.kernels
            .iter()
            .find(|entry| entry.name == "divide_two")
            .unwrap()
            .fallible
    );
    assert!(
        abi.kernels
            .iter()
            .find(|entry| entry.name == "divide_zero")
            .unwrap()
            .fallible
    );
    let module = support::artifact::module(&device, &directory)?;
    for name in ["divide_two", "divide_zero"] {
        let pipeline = Pipeline::new(&device, &module, name)?;
        let mut output = device.buffer_from_slice(&[99_u32; 3])?;
        // SAFETY: the safe kernel gives each thread at most one output element.
        let result = unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<32>::for_elements(3)?,
                (&mut output, 10_u32),
                |(output, value)| Ok([Argument::write(output), Argument::value(*value)?]),
            )
        }
        .wait();
        if name == "divide_two" {
            result?;
            assert_eq!(output.read().wait(), [5; 3]);
        } else {
            let Err(metal_oxide::Error::KernelFailed(failure)) = result else {
                panic!("expected division to fail");
            };
            assert_eq!(failure.kernel, "divide_zero");
            assert_eq!(failure.message, "attempt to divide by zero");
            assert!(failure.file.ends_with("generic_checks.rs"));
            assert_eq!(failure.line, 9);
            assert_eq!(output.read().wait(), [99; 3]);
        }
    }
    Ok(())
}
