use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn iterator_adapters_match_the_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/adapters.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;

    let n = 512;
    let a = (0..n).map(|i| (i % 7) as f32).collect::<Vec<_>>();
    let b = (0..n).map(|i| (i % 3 + 1) as f32).collect::<Vec<_>>();
    let (a_buffer, b_buffer) = (device.buffer_from_slice(&a)?, device.buffer_from_slice(&b)?);
    let mut out = device.buffer_zeroed::<f32>(n)?;
    let pipeline = Pipeline::new(&device, &module, "dot")?;
    // SAFETY: the grid covers every element; each thread writes only its own.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::for_elements(n as u32)?,
            (&a_buffer, &b_buffer, &mut out),
            |(a, b, out)| Ok([Argument::read(a), Argument::read(b), Argument::write(out)]),
        )
        .wait()?;
    }
    let expected = (0..n)
        .map(|i| a.iter().skip(i).zip(&b).map(|(x, y)| x * y).sum::<f32>())
        .collect::<Vec<_>>();
    assert_eq!(out.read().wait(), expected);

    let values = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0])?;
    let mut out = device.buffer_zeroed::<f32>(10)?;
    let pipeline = Pipeline::new(&device, &module, "adapters")?;
    // SAFETY: one thread writes the ten values.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&values, &mut out),
            |(values, out)| {
                Ok([
                    Argument::read(values),
                    Argument::write(out),
                    Argument::value(2.0_f32)?,
                ])
            },
        )
        .wait()?;
    }
    assert_eq!(
        out.read().wait(),
        [40.0, 129.0, 40.0, 3.0, 2.0, 1.0, 9.0, 14.0, 5.0, 5.0]
    );

    let mut out = device.buffer_zeroed::<f32>(4)?;
    let pipeline = Pipeline::new(&device, &module, "mutation")?;
    // SAFETY: one thread writes the four values.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            &mut out,
            |out| Ok([Argument::write(out), Argument::value(2.0_f32)?]),
        )
        .wait()?;
    }
    assert_eq!(out.read().wait(), [11.5, 8.5, 5.5, 2.5]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn raw_pointer_and_inclusive_range_kernels_match_the_cpu() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/raw_pointers.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let values = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0])?;
    let mut out = device.buffer_zeroed::<f32>(4)?;
    let pipeline = Pipeline::new(&device, &module, "raw_pointers")?;
    // SAFETY: one thread writes the four values.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (&values, &mut out),
            |(values, out)| {
                Ok([
                    Argument::read(values),
                    Argument::write(out),
                    Argument::value(7_i32)?,
                ])
            },
        )
        .wait()?;
    }
    assert_eq!(out.read().wait(), [75.0, 80.0, 5.0, -2.0]);

    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/range_inclusive.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    let mut out = device.buffer_zeroed::<u32>(1)?;
    let pipeline = Pipeline::new(&device, &module, "range_inclusive")?;
    // SAFETY: one thread writes its element.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            &mut out,
            |out| Ok([Argument::write(out), Argument::value(4_u32)?]),
        )
        .wait()?;
    }
    assert_eq!(out.read().wait(), [10]);
    Ok(())
}
