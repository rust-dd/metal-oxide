use super::*;
use metal_oxide::{Error, GpuValue};

#[path = "../core_methods/model.rs"]
mod model;

fn module(device: &Device) -> metal_oxide::Result<Module> {
    let directory = emitted(model::FIXTURE, metal_oxide_artifact::MathMode::Precise);
    support::artifact::module(device, &directory)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn core_integer_methods_match_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let powers = Pipeline::new(&device, &module, "powers")?;
    let wide = Pipeline::new(&device, &module, "wide_powers")?;
    let cmp = Pipeline::new(&device, &module, "comparisons")?;
    let division = Pipeline::new(&device, &module, "checked_division")?;
    let bits = Pipeline::new(&device, &module, "bit_methods")?;
    let primitive = Pipeline::new(&device, &module, "primitive_ordering")?;
    let config = LaunchConfig::<1>::new(Dim3::x(1));
    for &(a, b, exponent) in model::POWERS {
        let mut output = device.buffer_from_slice(&[123_u32; 19])?;
        // SAFETY: one thread writes eighteen elements, leaving the final guard unchanged.
        unsafe {
            Launch::new(
                Ok(&powers),
                config,
                (&mut output, a, b, exponent),
                |(out, a, b, exp)| {
                    Ok([
                        Argument::write(out),
                        Argument::value(*a)?,
                        Argument::value(*b)?,
                        Argument::value(*exp)?,
                    ])
                },
            )
        }
        .wait()?;
        let actual = output.read().wait();
        assert_eq!(
            &actual[..18],
            &model::powers(a, b, exponent),
            "{a}, {b}, {exponent}"
        );
        assert_eq!(actual[18], 123);
    }
    for &(a, exponent) in model::WIDE_POWERS {
        let mut output = device.buffer_from_slice(&[123_u64; 6])?;
        // SAFETY: one thread owns five output elements, followed by a guard.
        unsafe {
            Launch::new(
                Ok(&wide),
                config,
                (&mut output, a, exponent),
                |(out, a, exp)| {
                    Ok([
                        Argument::write(out),
                        Argument::value(*a)?,
                        Argument::value(*exp)?,
                    ])
                },
            )
        }
        .wait()?;
        let actual = output.read().wait();
        assert_eq!(
            &actual[..5],
            &model::wide_powers(a, exponent),
            "{a}, {exponent}"
        );
        assert_eq!(actual[5], 123);
    }
    for &(a, b, c, d) in model::INTEGERS {
        let mut output = device.buffer_from_slice(&[123_i32; 8])?;
        // SAFETY: one thread writes seven initialized output elements.
        unsafe {
            Launch::new(
                Ok(&cmp),
                config,
                (&mut output, a, b, c, d),
                |(out, a, b, c, d)| {
                    Ok([
                        Argument::write(out),
                        Argument::value(*a)?,
                        Argument::value(*b)?,
                        Argument::value(*c)?,
                        Argument::value(*d)?,
                    ])
                },
            )
        }
        .wait()?;
        let actual = output.read().wait();
        assert_eq!(
            &actual[..7],
            &model::comparisons(a, b, c, d),
            "{a}, {b}, {c}, {d}"
        );
        assert_eq!(actual[7], 123);
        let mut output = device.buffer_from_slice(&[123_u32; 9])?;
        // SAFETY: one thread writes eight output elements, followed by a guard.
        unsafe {
            Launch::new(
                Ok(&division),
                config,
                (&mut output, a, b, c, d),
                |(out, a, b, c, d)| {
                    Ok([
                        Argument::write(out),
                        Argument::value(*a)?,
                        Argument::value(*b)?,
                        Argument::value(*c)?,
                        Argument::value(*d)?,
                    ])
                },
            )
        }
        .wait()?;
        let actual = output.read().wait();
        assert_eq!(
            &actual[..8],
            &model::checked_division(a, b, c, d),
            "{a}, {b}, {c}, {d}"
        );
        assert_eq!(actual[8], 123);
    }
    for &value in model::BITS {
        let mut output = device.buffer_from_slice(&[123_u64; 12])?;
        // SAFETY: one thread writes eleven output elements, followed by a guard.
        unsafe {
            Launch::new(Ok(&bits), config, (&mut output, value), |(out, value)| {
                Ok([Argument::write(out), Argument::value(*value)?])
            })
        }
        .wait()?;
        let actual = output.read().wait();
        assert_eq!(&actual[..11], &model::bit_methods(value), "{value:x}");
        assert_eq!(actual[11], 123);
    }
    for &a in model::BITS {
        for &b in model::BITS {
            let mut output = device.buffer_from_slice(&[123_i32; 11])?;
            // SAFETY: one thread writes ten output elements, followed by a guard.
            unsafe {
                Launch::new(
                    Ok(&primitive),
                    config,
                    (&mut output, a, b),
                    |(out, a, b)| {
                        Ok([
                            Argument::write(out),
                            Argument::value(*a)?,
                            Argument::value(*b)?,
                        ])
                    },
                )
            }
            .wait()?;
            let actual = output.read().wait();
            assert_eq!(
                &actual[..10],
                &model::primitive_ordering(a, b),
                "{a:x}, {b:x}"
            );
            assert_eq!(actual[10], 123);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn core_float_signs_and_ordering_match_cpu_bits() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let pipeline = Pipeline::new(&device, &module, "float_methods")?;
    for &a in model::FLOATS {
        for &b in model::FLOATS {
            let mut output = device.buffer_from_slice(&[123_u32; 5])?;
            // SAFETY: one thread writes four output elements, followed by a guard.
            unsafe {
                Launch::new(
                    Ok(&pipeline),
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    (&mut output, f32::from_bits(a), f32::from_bits(b)),
                    |(out, a, b)| {
                        Ok([
                            Argument::write(out),
                            Argument::value(*a)?,
                            Argument::value(*b)?,
                        ])
                    },
                )
            }
            .wait()?;
            let actual = output.read().wait();
            assert_eq!(
                &actual[..4],
                &model::float_methods(a, b),
                "a={a:08x}, b={b:08x}"
            );
            assert_eq!(actual[4], 123);
        }
    }
    let bits = Pipeline::new(&device, &module, "float_bit_methods")?;
    for &a in model::SUBNORMALS {
        for &b in model::FLOATS {
            let mut output = device.buffer_from_slice(&[123_u32; 4])?;
            // SAFETY: one thread writes three output elements, followed by a guard.
            unsafe {
                Launch::new(
                    Ok(&bits),
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    (&mut output, f32::from_bits(a), f32::from_bits(b)),
                    |(out, a, b)| {
                        Ok([
                            Argument::write(out),
                            Argument::value(*a)?,
                            Argument::value(*b)?,
                        ])
                    },
                )
            }
            .wait()?;
            let actual = output.read().wait();
            assert_eq!(
                &actual[..3],
                &model::float_methods(a, b)[..3],
                "a={a:08x}, b={b:08x}"
            );
            assert_eq!(actual[3], 123);
        }
    }
    Ok(())
}

fn clamp<T: GpuValue>(
    device: &Device,
    pipeline: &Pipeline,
    values: (T, T, T),
    sentinel: T,
) -> metal_oxide::Result<(metal_oxide::Result<()>, Vec<T>)> {
    let mut output = device.buffer_from_slice(&[sentinel; 2])?;
    // SAFETY: one thread writes the first output element, leaving the final guard unchanged.
    let result = unsafe {
        Launch::new(
            Ok(pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&mut output, values),
            |(out, (value, low, high))| {
                Ok([
                    Argument::write(out),
                    Argument::value(*value)?,
                    Argument::value(*low)?,
                    Argument::value(*high)?,
                ])
            },
        )
    }
    .wait()
    .map(|_| ());
    Ok((result, output.read().wait().to_vec()))
}

fn failure(result: metal_oxide::Result<()>, kernel: &str, message: &str) {
    let Err(Error::KernelFailed(failure)) = result else {
        panic!("expected {kernel} failure, got {result:?}");
    };
    assert_eq!(failure.kernel, kernel);
    assert_eq!(failure.message, message);
    let source = if kernel == "float_clamp" {
        "core/src/num/f32.rs"
    } else {
        "fixtures/core_methods.rs"
    };
    assert!(failure.file.ends_with(source), "{failure}");
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn core_clamps_match_cpu_and_invalid_bounds_fail_before_storing() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = module(&device)?;
    let float = Pipeline::new(&device, &module, "float_clamp")?;
    let unsigned = Pipeline::new(&device, &module, "unsigned_clamp")?;
    let signed = Pipeline::new(&device, &module, "signed_clamp")?;
    for &(value, low, high) in model::FLOAT_CLAMPS {
        let (result, output) = clamp(&device, &float, (value, low, high), 123.0_f32)?;
        if low <= high {
            result?;
            assert_eq!(output[0].to_bits(), value.clamp(low, high).to_bits());
        } else {
            failure(result, "float_clamp", "min > max, or either was NaN");
            assert_eq!(output[0], 123.0);
        }
        assert_eq!(output[1], 123.0);
    }
    for &(value, low, high) in model::UNSIGNED_CLAMPS {
        let (result, output) = clamp(&device, &unsigned, (value, low, high), 123_u32)?;
        if low <= high {
            result?;
            assert_eq!(output[0], value.clamp(low, high));
        } else {
            failure(result, "unsigned_clamp", "min > max");
            assert_eq!(output[0], 123);
        }
        assert_eq!(output[1], 123);
    }
    for &(value, low, high) in model::SIGNED_CLAMPS {
        let (result, output) = clamp(&device, &signed, (value, low, high), 123_i32)?;
        if low <= high {
            result?;
            assert_eq!(output[0], value.clamp(low, high));
        } else {
            failure(result, "signed_clamp", "min > max");
            assert_eq!(output[0], 123);
        }
        assert_eq!(output[1], 123);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn core_power_with_overflow_checks_fails_before_storing() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/core_pow_checked.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let pipeline = Pipeline::new(&device, &module, "power")?;
    for exponent in [0_u32, 30, 31, 32, u32::MAX] {
        let mut output = device.buffer_from_slice(&[123_u32; 2])?;
        // SAFETY: one thread owns the first output element and receives u32 scalar arguments.
        let result = unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut output, exponent),
                |(out, exponent)| {
                    Ok([
                        Argument::write(out),
                        Argument::value(2_u32)?,
                        Argument::value(*exponent)?,
                    ])
                },
            )
        }
        .wait();
        if let Some(expected) = 2_u32.checked_pow(exponent) {
            result?;
            assert_eq!(output.read().wait(), [expected, 123]);
        } else {
            let Err(Error::KernelFailed(failure)) = result else {
                panic!("expected power overflow, got {result:?}");
            };
            assert_eq!(failure.kernel, "power");
            assert!(failure.message.contains("overflow"), "{failure}");
            assert_eq!(output.read().wait(), [123, 123]);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn core_branch_hints_keep_checked_division_conditions() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/core_hints.rs",
        "hints",
    )?;
    for divisor in [0_u32, 1, 3, u32::MAX] {
        let mut output = device.buffer_from_slice(&[123_u32; 4])?;
        // SAFETY: one thread writes three output elements, followed by a guard.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut output, divisor),
                |(out, divisor)| {
                    Ok([
                        Argument::write(out),
                        Argument::value(7_u32)?,
                        Argument::value(*divisor)?,
                    ])
                },
            )
        }
        .wait()?;
        assert_eq!(
            output.read().wait(),
            [
                0,
                7_u32.checked_div(divisor).unwrap_or(0),
                7_u32.checked_rem(divisor).unwrap_or(0),
                123
            ]
        );
    }
    Ok(())
}
