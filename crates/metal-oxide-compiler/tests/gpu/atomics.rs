use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn relaxed_atomics_preserve_old_values_cas_results_and_wrapping() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let source = "crates/metal-oxide-compiler/tests/fixtures/atomic_ops.rs";
    for space in ["atomic", "shared"] {
        let unsigned = pipeline(&device, source, &format!("{space}_u32"))?;
        let signed = pipeline(&device, source, &format!("{space}_i32"))?;
        let first = usize::from(space == "shared");
        let mut cell = device.buffer_zeroed::<u32>(1)?;
        let mut out = device.buffer_zeroed::<u32>(16)?;
        // SAFETY: one thread operates on one initialized cell and sixteen separate output elements.
        unsafe {
            device.launch(
                &unsigned,
                LaunchConfig::<1>::new(Dim3::x(1)),
                &[
                    Argument::atomic(&mut cell),
                    Argument::write(&mut out),
                    Argument::value(u32::MAX)?,
                ][first..],
            )?;
        }
        assert_eq!(
            &out.as_slice()[..11],
            &[u32::MAX, u32::MAX, 5, 8, 7, 2, 9, 1, 5, 6, 0]
        );
        assert_eq!(out.as_slice()[11], 6);
        let success = out.as_slice()[12];
        assert!(success <= 1);
        assert_eq!(out.as_slice()[13], if success == 1 { 15 } else { 6 });
        assert_eq!(&out.as_slice()[14..], &[u32::MAX, 0]);
        let mut cell = device.buffer_zeroed::<i32>(1)?;
        let mut out = device.buffer_zeroed::<i32>(16)?;
        // SAFETY: the signed kernel has the same single-thread resource contract.
        unsafe {
            device.launch(
                &signed,
                LaunchConfig::<1>::new(Dim3::x(1)),
                &[
                    Argument::atomic(&mut cell),
                    Argument::write(&mut out),
                    Argument::value(i32::MAX)?,
                ][first..],
            )?;
        }
        assert_eq!(
            &out.as_slice()[..11],
            &[i32::MAX, i32::MAX, 5, 8, 7, -2, 9, 1, 5, 6, 0]
        );
        assert_eq!(out.as_slice()[11], 6);
        let success = out.as_slice()[12];
        assert!(success == 0 || success == 1);
        assert_eq!(out.as_slice()[13], if success == 1 { 15 } else { 6 });
        assert_eq!(&out.as_slice()[14..], &[i32::MAX, i32::MIN]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn threadgroup_atomics_count_contention_in_separate_blocks() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(
        &device,
        "crates/metal-oxide-compiler/tests/fixtures/atomic_ops.rs",
        "shared_counter",
    )?;
    let mut output = device.buffer_zeroed::<u32>(514)?;
    // SAFETY: 257 complete blocks initialize two atomic cells before contended access.
    unsafe {
        device.launch(
            &pipeline,
            LaunchConfig::<256>::new(Dim3::x(257)),
            &[Argument::write(&mut output)],
        )?;
    }
    assert!(output.as_slice().iter().all(|value| *value == 128));
    Ok(())
}
