use super::*;
use metal_oxide::F16;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn half_storage_preserves_all_patterns_and_widens_exactly() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/half.rs",
        "half_bits",
    )?;
    let values = (0..=u16::MAX).map(F16::from_bits).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut bits = device.buffer_zeroed::<u16>(values.len())?;
    let mut wide = device.buffer_zeroed::<f32>(values.len())?;
    let mut copied = device.buffer_zeroed::<F16>(values.len())?;
    // SAFETY: buffers cover every binary16 pattern; each global thread owns its output index.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(values.len() as u32)?,
            (
                &input,
                &mut bits,
                &mut wide,
                &mut copied,
                values.len() as u32,
            ),
            |(input, bits, wide, copied, v4)| {
                Ok([
                    Argument::read(input),
                    Argument::write(bits),
                    Argument::write(wide),
                    Argument::write(copied),
                    Argument::value(*v4)?,
                ])
            },
        )
    }
    .wait()?;
    for (index, &value) in values.iter().enumerate() {
        assert_eq!(bits.read().wait()[index], index as u16);
        assert_eq!(copied.read().wait()[index].to_bits(), index as u16);
        let actual = wide.read().wait()[index];
        let expected = value.to_f32();
        if expected.is_nan() {
            assert!(actual.is_nan(), "half {index:04x}");
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "half {index:04x}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn half_conversion_rounds_ties_to_even_and_preserves_special_values() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/half.rs",
        "half_round",
    )?;
    let mut values = vec![
        0.0_f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0,
        65520.0,
        -65520.0,
    ];
    for bits in 0..0x7c00_u16 {
        let a = F16::from_bits(bits).to_f32();
        let b = F16::from_bits(bits + 1).to_f32();
        let midpoint = (a + b) * 0.5;
        if midpoint.is_finite() {
            values.extend([
                midpoint,
                -midpoint,
                f32::from_bits(midpoint.to_bits().saturating_sub(1)),
                f32::from_bits(midpoint.to_bits() + 1),
            ]);
        }
    }
    values.extend((0..4096_u32).map(|i| f32::from_bits(i.wrapping_mul(0x9e3779b9))));
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<F16>(values.len())?;
    // SAFETY: each global thread converts one initialized f32 into its own full-sized output slot.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(values.len() as u32)?,
            (&input, &mut output, values.len() as u32),
            |(input, output, v2)| {
                Ok([
                    Argument::read(input),
                    Argument::write(output),
                    Argument::value(*v2)?,
                ])
            },
        )
    }
    .wait()?;
    for (index, &value) in values.iter().enumerate() {
        let actual = output.read().wait()[index];
        let expected = F16::from_f32(value);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "f32 {:08x}",
                value.to_bits()
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn half_record_parameters_and_ctfe_constants_match_their_layout() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/half.rs",
        "half_record",
    )?;
    record!(HalfRecord((u8, (F16, F16), [F16; 2])) { tag, pair, samples });
    let value = HalfRecord((
        7_u8,
        (F16::from_f32(1.0), F16::from_f32(-1.0)),
        [F16::from_bits(1), F16::INFINITY],
    ));
    let mut output = device.buffer_zeroed::<f32>(6)?;
    // SAFETY: the value has the kernel's record layout; one thread owns six output elements.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&mut output, value, F16::from_f32(2.0)),
            |(output, value, v2)| {
                Ok([
                    Argument::write(output),
                    Argument::value(*value)?,
                    Argument::value(*v2)?,
                ])
            },
        )
    }
    .wait()?;
    let expected = [
        2.0,
        -1.0,
        F16::from_bits(1).to_f32(),
        f32::INFINITY,
        1.0,
        F16::from_bits(0x8001).to_f32(),
    ];
    for (&actual, expected) in output.read().wait().iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
    Ok(())
}
