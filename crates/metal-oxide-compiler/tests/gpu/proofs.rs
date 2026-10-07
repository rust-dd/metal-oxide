use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn checked_arithmetic_indices_and_division_match_rust() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_good.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    let module = Module::from_source(
        &device,
        &std::fs::read_to_string(directory.join("kernels.metal"))?,
    )?;
    let division = Pipeline::new(&device, &module, "guarded_division")?;
    let range = Pipeline::new(&device, &module, "checked_range")?;
    let arrays = Pipeline::new(&device, &module, "checked_array_loop")?;
    let index = Pipeline::new(&device, &module, "guarded_index")?;
    let narrow = Pipeline::new(&device, &module, "narrow_division")?;
    let config = LaunchConfig::<1>::new(Dim3::x(1));
    for (value, divisor) in [
        (i32::MIN, -1),
        (i32::MIN, 4),
        (i32::MAX, -7),
        (-123, -7),
        (123, 7),
        (123, 0),
    ] {
        let mut output = device.buffer_from_slice(&[987_i32; 3])?;
        // SAFETY: one thread owns all three output elements; scalar types match their bindings.
        unsafe {
            device.launch(
                &division,
                config,
                &[
                    Argument::write(&mut output),
                    Argument::value(value)?,
                    Argument::value(divisor)?,
                ],
            )?;
        }
        let expected = if divisor == 0 || (value == i32::MIN && divisor == -1) {
            [987; 3]
        } else {
            [value / divisor, value % divisor, value / 4 + 1]
        };
        assert_eq!(output.as_slice(), expected);
    }
    for value in [0_u32, 1, 7, 8, 15, 16, u32::MAX] {
        let mut output = device.buffer_from_slice(&[987_u32])?;
        // SAFETY: the sole thread owns the single output element.
        unsafe {
            device.launch(
                &range,
                config,
                &[Argument::write(&mut output), Argument::value(value)?],
            )?;
        }
        let expected = if value >= 16 {
            987
        } else if value < 8 {
            (value + 1) * 2
        } else {
            value + 9
        };
        assert_eq!(output.as_slice(), [expected]);
    }
    let mut output = device.buffer_zeroed::<u32>(4)?;
    // SAFETY: the sole thread writes the four-element output after updating its private array.
    unsafe {
        device.launch(&arrays, config, &[Argument::write(&mut output)])?;
    }
    assert_eq!(output.as_slice(), [11, 21, 31, 41]);
    for value in [0_u32, 1, 2, 3, 4, u32::MAX] {
        let mut output = device.buffer_from_slice(&[987_u32])?;
        // SAFETY: the sole thread writes one output; the kernel guards its private array index.
        unsafe {
            device.launch(
                &index,
                config,
                &[Argument::write(&mut output), Argument::value(value)?],
            )?;
        }
        assert_eq!(
            output.as_slice(),
            [if value < 4 { 11 + value * 10 } else { 987 }]
        );
    }
    for (a, b) in [
        (i8::MIN, i16::MIN),
        (-1, -1),
        (0, 0),
        (1, 1),
        (i8::MAX, i16::MAX),
    ] {
        let mut out8 = device.buffer_zeroed::<i8>(2)?;
        let mut out16 = device.buffer_zeroed::<i16>(2)?;
        // SAFETY: the sole thread owns both two-element outputs, whose element types match the kernel.
        unsafe {
            device.launch(
                &narrow,
                config,
                &[
                    Argument::write(&mut out8),
                    Argument::write(&mut out16),
                    Argument::value(a)?,
                    Argument::value(b)?,
                ],
            )?;
        }
        assert_eq!(out8.as_slice(), [a / 3, a % 3]);
        assert_eq!(out16.as_slice(), [b / 3, b % 3]);
    }
    Ok(())
}
