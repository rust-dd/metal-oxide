use super::*;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/transcendentals.rs";
const SENTINEL: f32 = -8192.0;

fn reference(device: &Device, entry: &str) -> metal_oxide::Result<Pipeline> {
    let module = Module::from_source(device, include_str!("../fixtures/transcendentals.metal"))?;
    Pipeline::new(device, &module, entry)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn precise_transcendentals_match_cpu_bounds_and_reference_msl() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let generated = pipeline(&device, FIXTURE, "transcendentals")?;
    let reference = reference(&device, "transcendentals")?;
    let cases = cases();
    let x = cases.iter().map(|&(x, _)| x).collect::<Vec<_>>();
    let y = cases.iter().map(|&(_, y)| y).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&x)?;
    let exponent = device.buffer_from_slice(&y)?;
    for n in [0, 1, 257, cases.len() as u32] {
        let mut results = Vec::new();
        for pipeline in [&generated, &reference] {
            let mut output = device.buffer_from_slice(&vec![SENTINEL; (n as usize + 32) * 8])?;
            // SAFETY: inputs cover n; disjoint output has eight elements per active thread.
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
            let output = output.as_slice();
            assert!(output[n as usize * 8..].iter().all(|&v| v == SENTINEL));
            results.push(output[..n as usize * 8].to_vec());
        }
        for (i, (&actual, &msl)) in results[0].iter().zip(&results[1]).enumerate() {
            assert!(
                same_bits_or_nan(actual, msl),
                "reference MSL at {i}: {actual} != {msl}"
            );
        }
        let mut failures = Vec::new();
        for (index, actual) in results[0].as_chunks::<8>().0.iter().enumerate() {
            let (x, y) = cases[index];
            let expected = values(x, y);
            let flushed = values(flush(x), flush(y));
            for (i, (operation, ulps)) in [
                ("ln", 4),
                ("exp", 4),
                ("sin", 4),
                ("cos", 4),
                ("powf", 16),
                ("tanh", 5),
                ("atan", 5),
                ("floor", 0),
            ]
            .into_iter()
            .enumerate()
            {
                if !matches(actual[i], expected[i], ulps) && !matches(actual[i], flushed[i], ulps) {
                    failures.push(format!(
                        "{operation}({x:?}, {y:?}): {:?}, expected {:?} or {:?}",
                        actual[i], expected[i], flushed[i]
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn sin_cos_matches_separate_sin_and_cos_in_both_modes() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let mut x = cases().into_iter().map(|(x, _)| x).collect::<Vec<_>>();
    // Strided bit patterns cover every exponent, both signs, NaNs and subnormals.
    x.extend((0..1 << 16).map(|k: u32| f32::from_bits(k.wrapping_mul(65_537) ^ 0x5a5a)));
    let n = x.len() as u32;
    let input = device.buffer_from_slice(&x)?;
    for mode in [
        metal_oxide_artifact::MathMode::Precise,
        metal_oxide_artifact::MathMode::Fast,
    ] {
        let pairs = pipeline_with_math_mode(
            &device,
            "crates/metal-oxide-compiler/tests/fixtures/sin_cos.rs",
            "sin_cos_pairs",
            mode,
        )?;
        let mut out = device.buffer_from_slice(&vec![SENTINEL; 4 * x.len() + 4])?;
        // SAFETY: each active lane writes its own four outputs; the last four stay untouched.
        unsafe {
            device.launch(
                &pairs,
                LaunchConfig::<64>::for_elements(n)?,
                &[
                    Argument::read(&input),
                    Argument::write(&mut out),
                    Argument::value(n)?,
                ],
            )?;
        }
        let (values, padding) = out.as_slice().split_at(4 * x.len());
        assert!(padding.iter().all(|&v| v == SENTINEL));
        for (value, out) in x.iter().zip(values.as_chunks::<4>().0) {
            assert!(
                same_bits_or_nan(out[0], out[2]) && same_bits_or_nan(out[1], out[3]),
                "{mode:?} sin_cos({value:e}) = ({:e}, {:e}), separately ({:e}, {:e})",
                out[0],
                out[1],
                out[2],
                out[3]
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn box_muller_euler_step_matches_cpu_and_reference_msl() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let generated = pipeline(&device, FIXTURE, "box_muller_step")?;
    let reference = reference(&device, "box_muller_step")?;
    for n in [0_u32, 1, 255, 256, 257, 4097] {
        let u = (0..n)
            .map(|i| (i % 1024 + 1) as f32 / 1024.0)
            .collect::<Vec<_>>();
        let v = (0..n).map(|i| (i % 997) as f32 / 997.0).collect::<Vec<_>>();
        let input_u = device.buffer_from_slice(&u)?;
        let input_v = device.buffer_from_slice(&v)?;
        let mut results = Vec::new();
        for pipeline in [&generated, &reference] {
            let mut output = device.buffer_from_slice(&vec![SENTINEL; n as usize + 128])?;
            // SAFETY: disjoint buffers cover n; the guarded 1D kernel owns each output index.
            unsafe {
                device.launch(
                    pipeline,
                    LaunchConfig::<128>::for_elements(n)?,
                    &[
                        Argument::read(&input_u),
                        Argument::read(&input_v),
                        Argument::write(&mut output),
                        Argument::value(n)?,
                    ],
                )?;
            }
            let output = output.as_slice();
            assert!(output[n as usize..].iter().all(|&v| v == SENTINEL));
            results.push(output[..n as usize].to_vec());
        }
        assert_eq!(results[0], results[1]);
        for (i, &actual) in results[0].iter().enumerate() {
            let z = (-2.0 * u[i].ln()).sqrt() * (core::f32::consts::TAU * v[i]).cos();
            let expected = 100.0 * (0.01 + 0.2 * z).exp();
            assert!(
                (actual - expected).abs() <= 2e-5 * expected.abs(),
                "n={n}, i={i}: {actual} != {expected}"
            );
        }
    }
    Ok(())
}

fn cases() -> Vec<(f32, f32)> {
    let mut cases = vec![
        (-2.0, 3.0),
        (-2.0, 2.0),
        (-2.0, 0.5),
        (-1.0, f32::INFINITY),
        (0.0, 3.0),
        (-0.0, 3.0),
        (0.0, -3.0),
        (-0.0, -3.0),
        (0.0, 0.0),
        (f32::NAN, 0.0),
        (f32::NAN, 2.0),
        (1.0, f32::NAN),
        (2.0, f32::NAN),
        (f32::INFINITY, -2.0),
        (f32::NEG_INFINITY, 3.0),
        (f32::NEG_INFINITY, 0.5),
        (0.5, f32::INFINITY),
        (2.0, f32::NEG_INFINITY),
        (f32::MIN_POSITIVE, 1.0),
        (f32::from_bits(1), 1.0),
        (-f32::from_bits(1), 1.0),
        (f32::MAX, 1.0),
        (-f32::MAX, 1.0),
    ];
    cases.extend(
        [
            -104.0,
            -103.0,
            -100.0,
            -87.3,
            88.7,
            88.72,
            89.0,
            core::f32::consts::PI,
            -core::f32::consts::PI,
            core::f32::consts::FRAC_PI_2,
            -core::f32::consts::FRAC_PI_2,
            f32::from_bits(1.0_f32.to_bits() - 1),
            1.0 + f32::EPSILON,
        ]
        .into_iter()
        .map(|x| (x, 1.25)),
    );
    cases.extend((0..257).map(|i| ((i as f32 - 128.0) * 0.25, 1.25)));
    cases
}

fn values(x: f32, y: f32) -> [f32; 8] {
    let x = x as f64;
    let y = y as f64;
    [
        x.ln(),
        x.exp(),
        x.sin(),
        x.cos(),
        x.powf(y),
        x.tanh(),
        x.atan(),
        x.floor(),
    ]
    .map(|v| v as f32)
}

fn flush(value: f32) -> f32 {
    if value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x80000000)
    } else {
        value
    }
}

fn same_bits_or_nan(actual: f32, expected: f32) -> bool {
    actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan())
}

fn matches(actual: f32, expected: f32, ulps: u32) -> bool {
    if same_bits_or_nan(actual, expected) || same_bits_or_nan(actual, flush(expected)) {
        return true;
    }
    if !actual.is_finite() || !expected.is_finite() || actual == 0.0 || expected == 0.0 {
        return false;
    }
    ordered(actual).abs_diff(ordered(expected)) <= ulps
}

fn ordered(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x80000000 == 0 {
        bits | 0x80000000
    } else {
        !bits
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn rounding_matches_rust_in_precise_and_fast_modes() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let mut x = vec![
        -2.5_f32,
        -1.5,
        -0.5,
        -0.4,
        -0.0,
        0.0,
        0.4,
        0.5,
        1.5,
        2.5,
        8_388_607.5,
        -8_388_607.5,
        1e30,
        -1e30,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-40,
        -1e-40,
    ];
    x.extend((0..4096).map(|i| (i as f32 - 2048.0) * 0.37 + 0.13));
    let n = x.len() as u32;
    let input = device.buffer_from_slice(&x)?;
    for mode in [
        metal_oxide_artifact::MathMode::Precise,
        metal_oxide_artifact::MathMode::Fast,
    ] {
        let rounding = pipeline_with_math_mode(
            &device,
            "crates/metal-oxide-compiler/tests/fixtures/rounding.rs",
            "rounding",
            mode,
        )?;
        let mut out = device.buffer_from_slice(&vec![SENTINEL; 3 * x.len()])?;
        // SAFETY: each active lane writes its own three outputs.
        unsafe {
            device.launch(
                &rounding,
                LaunchConfig::<64>::for_elements(n)?,
                &[
                    Argument::read(&input),
                    Argument::write(&mut out),
                    Argument::value(n)?,
                ],
            )?;
        }
        let functions: [fn(f32) -> f32; 3] = [f32::round, f32::ceil, f32::trunc];
        for (i, &value) in x.iter().enumerate() {
            for (k, function) in functions.iter().enumerate() {
                let actual = out.as_slice()[3 * i + k];
                // Subnormal inputs may be flushed, and fast mode leaves zero signs open.
                let accepted = [function(value), function(flush(value))]
                    .iter()
                    .any(|&want| {
                        same_bits_or_nan(actual, want)
                            || (mode == metal_oxide_artifact::MathMode::Fast && actual == want)
                    });
                assert!(accepted, "{mode:?} function {k} of {value:e}: {actual:e}");
            }
        }
    }
    Ok(())
}
