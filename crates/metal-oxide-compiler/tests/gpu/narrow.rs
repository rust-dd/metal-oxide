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
                        Argument::u8(a),
                        Argument::u16(b),
                        Argument::u32(shift),
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
                Argument::u16(300),
                Argument::u32(256),
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
