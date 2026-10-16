use std::{fmt::Debug, sync::OnceLock};

use metal_oxide::{Argument, Device, GpuValue, Launch, LaunchConfig, Module, Operation, Pipeline};
use nostd_numeric_kernels::{float_values, integer_values, libm_values};

use super::{assert_float, float_inputs, kernels, support};

fn module(device: &Device) -> metal_oxide::Result<Module> {
    static ARTIFACT: OnceLock<(metal_oxide_artifact::Manifest, Vec<u8>)> = OnceLock::new();
    let (manifest, library) =
        ARTIFACT.get_or_init(|| support::artifact::manifest(kernels()).unwrap());
    eprintln!("no_std numeric workloads: {}", device.name());
    Module::from_bytes(device, &manifest.to_json()?, library)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn libm_f32_helpers_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let pipeline = Pipeline::new(&device, &module, "libm_f32")?;
    let values = float_inputs();
    for count in [0, 1, 257, values.len()] {
        let values = &values[..count];
        let input = device.buffer_from_slice(values)?;
        let mut output = device.buffer_zeroed::<[f32; 8]>(values.len())?;
        // SAFETY: equal-length, disjoint buffers satisfy the safe kernel's contract.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::for_elements(values.len() as u32)?,
                (&input, &mut output),
                |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
            )
        }
        .wait()?;
        for (&value, actual) in values.iter().zip(output.read().wait()) {
            for (operation, (actual, expected)) in actual.iter().zip(libm_values(value)).enumerate()
            {
                assert_float(
                    *actual,
                    expected,
                    &format!("libm({value:?}), op {operation}"),
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn generic_float_traits_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let pipeline = Pipeline::new(&device, &module, "traits_f32")?;
    let values = [
        -8.0, -4.0, -2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0,
    ];
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<[f32; 4]>(values.len())?;
    for exponent in [i32::MIN, -31, -7, -3, 0, 1, 2, 3, 8, 31, i32::MAX] {
        // SAFETY: equal-length, disjoint buffers satisfy the safe kernel's contract.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<32>::for_elements(values.len() as u32)?,
                (&input, &mut output, exponent),
                |(input, output, exponent)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(output),
                        Argument::value(*exponent)?,
                    ])
                },
            )
        }
        .wait()?;
        for (&value, actual) in values.iter().zip(output.read().wait()) {
            for (operation, (actual, expected)) in
                actual.iter().zip(float_values(value, exponent)).enumerate()
            {
                assert_float(
                    *actual,
                    expected,
                    &format!("Float({value:?}, {exponent}), op {operation}"),
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn generic_integer_traits_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    macro_rules! verify {
        ($($ty:ident),* $(,)?) => {$(
            let values = [
                <$ty>::MIN, 0, 1, 2, -3_i32 as $ty, -1_i32 as $ty,
                255_i32 as $ty, 256_i32 as $ty, 257_i32 as $ty,
                <$ty>::MAX / 2, <$ty>::MAX,
            ];
            integer::<$ty>(
                &device,
                &module,
                concat!("traits_", stringify!($ty)),
                &values,
                integer_values::<$ty>,
            )?;
        )*};
    }
    verify!(i8, i16, i32, u8, u16, u32, u64);
    Ok(())
}

fn integer<T: GpuValue + PartialEq + Debug>(
    device: &Device,
    module: &Module,
    entry: &str,
    values: &[T],
    expected: fn(T, u32) -> [T; 4],
) -> metal_oxide::Result<()> {
    let pipeline = Pipeline::new(device, module, entry)?;
    let input = device.buffer_from_slice(values)?;
    let mut output = device.buffer_zeroed::<[T; 4]>(values.len())?;
    for exponent in [0, 1, 2, 3, 7, 31, 32, 63, 64] {
        // SAFETY: equal-length, disjoint buffers satisfy the safe kernel's contract.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<32>::for_elements(values.len() as u32)?,
                (&input, &mut output, exponent),
                |(input, output, exponent)| {
                    Ok([
                        Argument::read(input),
                        Argument::write(output),
                        Argument::value(*exponent)?,
                    ])
                },
            )
        }
        .wait()?;
        for (&value, actual) in values.iter().zip(output.read().wait()) {
            assert_eq!(
                *actual,
                expected(value, exponent),
                "{entry}({value:?}, {exponent})"
            );
        }
    }
    Ok(())
}
