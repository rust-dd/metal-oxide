use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn float_negation_preserves_every_half_encoding_and_float_nan_payloads() -> metal_oxide::Result<()>
{
    let device = Device::system_default()?;
    let mut values = (0..=u16::MAX).map(u32::from).collect::<Vec<_>>();
    let mut bits = 0x9e37_79b9_u32;
    for _ in 0..8192 {
        bits = bits.wrapping_mul(1664525).wrapping_add(1013904223);
        values.push(bits);
        values.push(0x7f80_0000 | (bits & 0x807f_ffff));
    }
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<[u32; 2]>(values.len())?;
    for mode in [
        metal_oxide_artifact::MathMode::Precise,
        metal_oxide_artifact::MathMode::Fast,
    ] {
        let pipeline = pipeline_with_math_mode(
            &device,
            "crates/metal-oxide-compiler/tests/fixtures/float_negation.rs",
            "negate_bits",
            mode,
        )?;
        // SAFETY: each thread writes one output for its checked input index.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::for_elements(values.len() as u32)?,
                (&input, &mut output),
                |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
            )
        }
        .wait()?;
        for (&bits, result) in values.iter().zip(output.read().wait().iter()) {
            assert_eq!(
                *result,
                [bits ^ 0x8000_0000, (bits & 0xffff) ^ 0x8000],
                "{mode:?}: {bits:08x}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn float_casts_match_rust_at_boundaries_and_for_arbitrary_bits() -> metal_oxide::Result<()> {
    float_casts(metal_oxide_artifact::MathMode::Precise)
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn fast_math_preserves_saturating_casts_including_nan_and_infinity() -> metal_oxide::Result<()> {
    float_casts(metal_oxide_artifact::MathMode::Fast)
}

fn float_casts(math_mode: metal_oxide_artifact::MathMode) -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline_with_math_mode(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/float_casts.rs",
        "float_casts",
        math_mode,
    )?;
    let mut values = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        0.75,
        -0.75,
        1.75,
        -1.75,
    ];
    for boundary in [
        -2147483648.0_f32,
        -32768.0,
        -128.0,
        127.0,
        255.0,
        32767.0,
        65535.0,
        2147483648.0,
        4294967296.0,
        18446744073709551616.0,
    ] {
        values.extend([
            f32::from_bits(boundary.to_bits() - 1),
            boundary,
            f32::from_bits(boundary.to_bits() + 1),
        ]);
    }
    values.extend((0..1024_u32).map(|i| f32::from_bits(i.wrapping_mul(0x9e3779b9))));
    let n = values.len() as u32;
    let input = device.buffer_from_slice(&values)?;
    let mut i32s = device.buffer_zeroed::<i32>(values.len())?;
    let mut u32s = device.buffer_zeroed::<u32>(values.len())?;
    let mut i8s = device.buffer_zeroed::<i8>(values.len())?;
    let mut u8s = device.buffer_zeroed::<u8>(values.len())?;
    let mut i16s = device.buffer_zeroed::<i16>(values.len())?;
    let mut u16s = device.buffer_zeroed::<u16>(values.len())?;
    let mut high = device.buffer_zeroed::<u32>(values.len())?;
    // SAFETY: all buffers cover n elements; each active thread owns one output element.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<128>::for_elements(n)?,
            (
                &input, &mut i32s, &mut u32s, &mut i8s, &mut u8s, &mut i16s, &mut u16s, &mut high,
                n,
            ),
            |(input, i32s, u32s, i8s, u8s, i16s, u16s, high, n)| {
                Ok([
                    Argument::read(input),
                    Argument::write(i32s),
                    Argument::write(u32s),
                    Argument::write(i8s),
                    Argument::write(u8s),
                    Argument::write(i16s),
                    Argument::write(u16s),
                    Argument::write(high),
                    Argument::value(*n)?,
                ])
            },
        )
    }
    .wait()?;
    for (index, &value) in values.iter().enumerate() {
        let bits = value.to_bits();
        assert_eq!(i32s.read().wait()[index], value as i32, "i32 {bits:08x}");
        assert_eq!(u32s.read().wait()[index], value as u32, "u32 {bits:08x}");
        assert_eq!(i8s.read().wait()[index], value as i8, "i8 {bits:08x}");
        assert_eq!(u8s.read().wait()[index], value as u8, "u8 {bits:08x}");
        assert_eq!(i16s.read().wait()[index], value as i16, "i16 {bits:08x}");
        assert_eq!(u16s.read().wait()[index], value as u16, "u16 {bits:08x}");
        assert_eq!(
            high.read().wait()[index],
            ((value as usize) >> 32) as u32,
            "usize {bits:08x}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn precise_math_preserves_special_values_and_explicit_fusion() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/float_math.rs",
        "float_math",
    )?;
    let tiny = f32::from_bits(1);
    for (a, b, c) in [
        (1.0_f32 + f32::EPSILON, 1.0 - f32::EPSILON, -1.0),
        (2.0, 3.0, 4.0),
        (-4.0, -0.0, 1.0),
        (0.0, -0.0, 0.0),
        (-0.0, 0.0, -0.0),
        (f32::INFINITY, 1.0, -1.0),
        (f32::INFINITY, 0.0, 0.0),
        (f32::NAN, 3.0, 0.0),
        (3.0, f32::NAN, 0.0),
        (3.0, 4.0, f32::NAN),
        (f32::MIN_POSITIVE, f32::MIN_POSITIVE, 0.0),
        (tiny, 4.0, 0.0),
        (-tiny, 1.0, 0.0),
        (f32::MAX, 2.0, -f32::MAX),
        (f32::from_bits(0xff800123), 1.0, 0.0),
    ] {
        let mut output = device.buffer_zeroed::<f32>(7)?;
        // SAFETY: one thread owns the seven-element output; all scalar arguments are f32.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut output, a, b, c),
                |(output, a, b, c)| {
                    Ok([
                        Argument::write(output),
                        Argument::value(*a)?,
                        Argument::value(*b)?,
                        Argument::value(*c)?,
                    ])
                },
            )
        }
        .wait()?;
        let actual = output.read().wait();
        assert_eq!(actual[0].to_bits(), a.to_bits() & 0x7fffffff, "abs");
        assert_minmax(actual[1], a.min(b), flush(a).min(flush(b)));
        assert_minmax(actual[2], a.max(b), flush(a).max(flush(b)));
        assert_math(actual[3], a.sqrt(), flush(a).sqrt(), "sqrt");
        assert_math(
            actual[4],
            a.mul_add(b, c),
            flush(a).mul_add(flush(b), flush(c)),
            "fma",
        );
        assert_math(
            actual[5],
            a * b + c,
            flush(a) * flush(b) + flush(c),
            "unfused",
        );
        assert_eq!(actual[6].to_bits(), a.to_bits(), "bit roundtrip");
    }
    Ok(())
}

fn flush(value: f32) -> f32 {
    if value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x80000000)
    } else {
        value
    }
}

fn matches(actual: f32, expected: f32) -> bool {
    if expected.is_nan() {
        actual.is_nan()
    } else {
        actual.to_bits() == expected.to_bits() || actual.to_bits() == flush(expected).to_bits()
    }
}

fn assert_math(actual: f32, expected: f32, with_flushed_inputs: f32, operation: &str) {
    assert!(
        matches(actual, expected) || matches(actual, with_flushed_inputs),
        "{operation}: {actual:?} expected {expected:?} or {with_flushed_inputs:?}"
    );
}

fn assert_minmax(actual: f32, expected: f32, with_flushed_inputs: f32) {
    if actual == 0.0 && (expected == 0.0 || with_flushed_inputs == 0.0) {
        return;
    }
    assert_math(actual, expected, with_flushed_inputs, "min/max");
}
