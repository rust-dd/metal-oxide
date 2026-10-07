use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn narrow_integer_math_preserves_rust_wrapping_and_shifts() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/narrow.rs",
        "narrow_math",
    )?;
    for (a, b) in [(0_u8, 0_u16), (1, 1), (128, 32768), (255, 65535)] {
        for shift in [0_u32, 7, 8, 15, 16, 31, 32, 63] {
            let mut out8 = device.buffer_zeroed::<u8>(6)?;
            let mut out16 = device.buffer_zeroed::<u16>(6)?;
            // SAFETY: one thread owns both six-element outputs; copied scalar arguments match their ABI.
            unsafe {
                device.launch(
                    &pipeline,
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    &[
                        Argument::write(&mut out8),
                        Argument::write(&mut out16),
                        Argument::value::<u8>(a)?,
                        Argument::value::<u16>(b)?,
                        Argument::value::<u32>(shift)?,
                    ],
                )?;
            }
            assert_eq!(
                out8.as_slice(),
                [
                    a.wrapping_add(255),
                    a.wrapping_sub(255),
                    a.wrapping_mul(255),
                    a.wrapping_shl(shift),
                    a.wrapping_shr(shift),
                    !a
                ]
            );
            assert_eq!(
                out16.as_slice(),
                [
                    b.wrapping_add(65535),
                    b.wrapping_sub(65535),
                    b.wrapping_mul(65535),
                    b.wrapping_shl(shift),
                    b.wrapping_shr(shift),
                    !b
                ]
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn byte_buffers_widen_and_scale_without_changing_stride() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/narrow.rs",
        "widen",
    )?;
    let values = (0_u8..=255).collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<u16>(256)?;
    // SAFETY: one 256-thread block owns the distinct full-sized output, and input covers every index.
    unsafe {
        device.launch(
            &pipeline,
            LaunchConfig::<256>::new(Dim3::x(1)),
            &[
                Argument::read(&input),
                Argument::write(&mut output),
                Argument::value::<u16>(300)?,
                Argument::value::<u32>(256)?,
            ],
        )?;
    }
    assert_eq!(
        output.as_slice(),
        values
            .iter()
            .map(|&v| (v as u16).wrapping_mul(300))
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn signed_narrow_math_and_flags_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let path = "crates/metal-oxide-compiler/tests/fixtures/signed_narrow.rs";
    let arithmetic = pipeline(&device, path, "signed_narrow")?;
    let flags = pipeline(&device, path, "signed_flags")?;
    let records = pipeline(&device, path, "signed_record")?;
    for (a, b) in [
        (i8::MIN, i16::MIN),
        (-127, -32767),
        (-1, -1),
        (0, 0),
        (1, 1),
        (126, 32766),
        (127, 32767),
    ] {
        for shift in [0, 7, 8, 15, 16, 31, 32, 63] {
            let mut out8 = device.buffer_zeroed::<i8>(8)?;
            let mut out16 = device.buffer_zeroed::<i16>(8)?;
            // SAFETY: one thread owns each eight-element output; argument types match the artifact.
            unsafe {
                device.launch(
                    &arithmetic,
                    LaunchConfig::<1>::new(Dim3::x(1)),
                    &[
                        Argument::write(&mut out8),
                        Argument::write(&mut out16),
                        Argument::value(a)?,
                        Argument::value(b)?,
                        Argument::value(shift)?,
                    ],
                )?;
            }
            assert_eq!(
                out8.as_slice(),
                [
                    a.wrapping_add(127),
                    a.wrapping_sub(127),
                    a.wrapping_mul(-127),
                    a.wrapping_shl(shift),
                    a.wrapping_shr(shift),
                    a.wrapping_neg(),
                    !a,
                    b as i8
                ]
            );
            assert_eq!(
                out16.as_slice(),
                [
                    b.wrapping_add(32767),
                    b.wrapping_sub(32767),
                    b.wrapping_mul(-32767),
                    b.wrapping_shl(shift),
                    b.wrapping_shr(shift),
                    b.wrapping_neg(),
                    !b,
                    a as i16
                ]
            );
        }
        let mut out8 = device.buffer_zeroed::<i8>(1)?;
        let mut out16 = device.buffer_zeroed::<i16>(1)?;
        let mut overflow = device.buffer_zeroed::<u32>(2)?;
        // SAFETY: one thread writes initialized values and flags in disjoint outputs.
        unsafe {
            device.launch(
                &flags,
                LaunchConfig::<1>::new(Dim3::x(1)),
                &[
                    Argument::write(&mut out8),
                    Argument::write(&mut out16),
                    Argument::write(&mut overflow),
                    Argument::value(a)?,
                    Argument::value(b)?,
                ],
            )?;
        }
        let (x, ox) = a.overflowing_mul(-127);
        let (y, oy) = b.overflowing_mul(-32767);
        assert_eq!(out8.as_slice(), [x]);
        assert_eq!(out16.as_slice(), [y]);
        assert_eq!(overflow.as_slice(), [ox as u32, oy as u32]);
        let mut record8 = device.buffer_zeroed::<i8>(2)?;
        let mut record16 = device.buffer_zeroed::<i16>(2)?;
        // SAFETY: the host tuple has the tested eight-byte record layout; one thread owns both outputs.
        unsafe {
            device.launch(
                &records,
                LaunchConfig::<1>::new(Dim3::x(1)),
                &[
                    Argument::write(&mut record8),
                    Argument::write(&mut record16),
                    Argument::value((a, b, (-1_i8, 7_u16)))?,
                ],
            )?;
        }
        let small = if b < 0 {
            a.wrapping_add(1)
        } else {
            a.wrapping_neg()
        };
        let wide = match b {
            i16::MIN => 127,
            -1 => 255,
            0 => 0,
            _ => a as u8 as i16,
        };
        assert_eq!(record8.as_slice(), [small, wide as i8]);
        assert_eq!(record16.as_slice(), [wide, 7]);
    }
    Ok(())
}
