use super::*;

const SOURCE: &str = "crates/metal-oxide-compiler/tests/fixtures/atomic_numeric.rs";

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn float_atomics_preserve_old_values_and_compare_bits() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let ops = pipeline(&device, SOURCE, "float_ops")?;
    let mut cell = device.buffer_zeroed::<f32>(1)?;
    let mut out = device.buffer_zeroed::<f32>(9)?;
    for start in [
        -0.0_f32,
        f32::INFINITY,
        f32::from_bits(0x7fc0_1234),
        -1024.0,
    ] {
        // SAFETY: a single thread accesses one atomic cell and nine output elements.
        unsafe {
            Launch::new(
                Ok(&ops),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut cell, &mut out, start),
                |(cell, out, start)| {
                    Ok([
                        Argument::atomic(cell),
                        Argument::write(out),
                        Argument::value(*start)?,
                    ])
                },
            )
        }
        .wait()?;
        let values = out.read().wait();
        assert_eq!(values[0].to_bits(), start.to_bits());
        assert_eq!(values[1].to_bits(), start.to_bits());
        assert_eq!(&values[2..7], &[5.0, 8.0, 7.0, 0.0, 7.0]);
        assert!(values[7] == 0.0 || values[7] == 1.0);
        assert_eq!(values[8], if values[7] == 1.0 { 15.0 } else { 7.0 });
    }
    let cas = pipeline(&device, SOURCE, "float_cas")?;
    let mut out = device.buffer_zeroed::<u32>(3)?;
    for (initial, expected) in [
        (0_u32, 0x8000_0000),
        (0x8000_0000, 0),
        (0x7fc0_1234, 0x7fc0_1234),
        (0x7fc0_1234, 0x7fc0_5678),
    ] {
        cell.write().wait()[0] = f32::from_bits(initial);
        // SAFETY: a single thread accesses one atomic cell and three output words.
        unsafe {
            Launch::new(
                Ok(&cas),
                LaunchConfig::<1>::new(Dim3::x(1)),
                (&mut cell, &mut out, f32::from_bits(expected), 9.0_f32),
                |(cell, out, expected, desired)| {
                    Ok([
                        Argument::atomic(cell),
                        Argument::write(out),
                        Argument::value(*expected)?,
                        Argument::value(*desired)?,
                    ])
                },
            )
        }
        .wait()?;
        let values = out.read().wait();
        assert_eq!(values[0], initial);
        assert!(values[1] <= 1);
        if initial != expected {
            assert_eq!(
                values[1], 0,
                "initial={initial:08x} expected={expected:08x}"
            );
        }
        assert_eq!(
            values[2],
            if values[1] == 1 {
                9.0_f32.to_bits()
            } else {
                initial
            }
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn float_and_u64_atomics_handle_contended_device_storage() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let floats = pipeline(&device, SOURCE, "float_contention")?;
    let wide = pipeline(&device, SOURCE, "wide_contention")?;
    for blocks in [1, 7, 257] {
        let mut cells = device.buffer_zeroed::<f32>(2)?;
        // SAFETY: every thread uses atomic operations on the two initialized cells.
        unsafe {
            Launch::new(
                Ok(&floats),
                LaunchConfig::<256>::new(Dim3::x(blocks)),
                (&mut cells,),
                |(cells,)| Ok([Argument::atomic(cells)]),
            )
        }
        .wait()?;
        assert_eq!(
            cells.read().wait(),
            [blocks as f32 * 256.0, -(blocks as f32) * 256.0]
        );
        let mut cells = device.buffer_from_slice(&[u64::MAX, 0])?;
        // SAFETY: every thread uses atomic operations on the two initialized cells.
        unsafe {
            Launch::new(
                Ok(&wide),
                LaunchConfig::<256>::new(Dim3::x(blocks)),
                (&mut cells,),
                |(cells,)| Ok([Argument::atomic(cells)]),
            )
        }
        .wait()?;
        assert_eq!(
            cells.read().wait(),
            [1_u64 << 48, (1_u64 << 48) + u64::from(blocks) * 256 - 1]
        );
    }
    Ok(())
}
