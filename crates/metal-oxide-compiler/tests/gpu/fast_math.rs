use super::*;
use metal_oxide_artifact::MathMode;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn fast_functions_match_reference_within_their_accuracy_domains() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let generated = pipeline_with_math_mode(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/transcendentals.rs",
        "transcendentals",
        MathMode::Fast,
    )?;
    let source = format!(
        "#define METAL_OXIDE_FAST_MATH\n{}",
        include_str!("../fixtures/transcendentals.metal")
    );
    let reference = Pipeline::new(
        &device,
        &Module::from_source(&device, &source)?,
        "transcendentals",
    )?;
    let values = (0..513)
        .map(|i| 0.125 + i as f32 / 256.0)
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let exponent = device.buffer_from_slice(&vec![1.25_f32; values.len()])?;
    for n in [0, 1, 257, values.len() as u32] {
        let mut results = Vec::new();
        for pipeline in [&generated, &reference] {
            let mut output = device.buffer_from_slice(&vec![-8192.0_f32; (n as usize + 32) * 8])?;
            // SAFETY: disjoint buffers cover n, with eight output values per guarded thread.
            unsafe {
                device.launch(
                    pipeline,
                    LaunchConfig::<32>::for_elements(n)?,
                    &[
                        Argument::read(&input),
                        Argument::read(&exponent),
                        Argument::write(&mut output),
                        Argument::value(n)?,
                    ],
                )?;
            }
            assert!(
                output.as_slice()[n as usize * 8..]
                    .iter()
                    .all(|&x| x == -8192.0)
            );
            results.push(output.as_slice()[..n as usize * 8].to_vec());
        }
        assert_eq!(results[0], results[1]);
        for (index, actual) in results[0].as_chunks::<8>().0.iter().enumerate() {
            let x = values[index] as f64;
            let expected = [
                x.ln(),
                x.exp(),
                x.sin(),
                x.cos(),
                x.powf(1.25),
                x.tanh(),
                x.atan(),
                x.floor(),
            ]
            .map(|v| v as f32);
            for (operation, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                let within = match operation {
                    0 if (0.5..=2.0).contains(&x) => (actual - expected).abs() <= 2.0_f32.powi(-21),
                    0 => actual.to_bits().abs_diff(expected.to_bits()) <= 3,
                    1 => {
                        actual.to_bits().abs_diff(expected.to_bits())
                            <= 3 + (2.0 * x).floor() as u32
                    }
                    2 | 3 => (actual - expected).abs() <= 2.0_f32.powi(-13),
                    4 | 5 => (actual - expected).abs() <= 2e-5 * expected.abs(),
                    6 => actual.to_bits().abs_diff(expected.to_bits()) <= 5,
                    7 => actual == expected,
                    _ => unreachable!(),
                };
                assert!(within, "operation {operation}({x}): {actual} != {expected}");
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn fast_source_contract_allows_fusion_and_keeps_nan_and_infinity() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline_with_math_mode(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/float_math.rs",
        "float_math",
        MathMode::Fast,
    )?;
    for (a, b, c) in [
        (1.0_f32 + f32::EPSILON, 1.0 - f32::EPSILON, -1.0),
        (f32::INFINITY, 1.0, 0.0),
        (f32::NAN, 3.0, 0.0),
        (3.0, f32::NAN, 0.0),
    ] {
        let mut output = device.buffer_zeroed::<f32>(7)?;
        // SAFETY: one thread owns all seven elements and each value argument is f32.
        unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<1>::new(Dim3::x(1)),
                &[
                    Argument::write(&mut output),
                    Argument::value(a)?,
                    Argument::value(b)?,
                    Argument::value(c)?,
                ],
            )?;
        }
        let actual = output.as_slice();
        let expected = a.mul_add(b, c);
        for &value in &actual[4..6] {
            assert!(
                value.to_bits() == expected.to_bits() || (value.is_nan() && expected.is_nan()),
                "{actual:?}"
            );
        }
        assert_eq!(actual[1], a.min(b));
        assert_eq!(actual[2], a.max(b));
        assert_eq!(actual[0].to_bits(), a.to_bits() & 0x7fffffff);
        assert_eq!(actual[6].to_bits(), a.to_bits());
    }
    Ok(())
}
