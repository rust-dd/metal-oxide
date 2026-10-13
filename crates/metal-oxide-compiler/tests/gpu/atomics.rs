use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn relaxed_atomics_preserve_old_values_cas_results_and_wrapping() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let source = "crates/metal-oxide-compiler/tests/fixtures/atomic_ops.rs";
    for space in ["atomic", "shared"] {
        let unsigned = pipeline(&device, source, &format!("{space}_u32"))?;
        let signed = pipeline(&device, source, &format!("{space}_i32"))?;
        let shared = space == "shared";
        let mut cell = device.buffer_zeroed::<u32>(1)?;
        let mut out = device.buffer_zeroed::<u32>(16)?;
        run(&unsigned, shared, &mut cell, &mut out, u32::MAX)?;
        assert_eq!(
            &out.read().wait()[..11],
            &[u32::MAX, u32::MAX, 5, 8, 7, 2, 9, 1, 5, 6, 0]
        );
        assert_eq!(out.read().wait()[11], 6);
        let success = out.read().wait()[12];
        assert!(success <= 1);
        assert_eq!(out.read().wait()[13], if success == 1 { 15 } else { 6 });
        assert_eq!(&out.read().wait()[14..], &[u32::MAX, 0]);
        let mut cell = device.buffer_zeroed::<i32>(1)?;
        let mut out = device.buffer_zeroed::<i32>(16)?;
        run(&signed, shared, &mut cell, &mut out, i32::MAX)?;
        assert_eq!(
            &out.read().wait()[..11],
            &[i32::MAX, i32::MAX, 5, 8, 7, -2, 9, 1, 5, 6, 0]
        );
        assert_eq!(out.read().wait()[11], 6);
        let success = out.read().wait()[12];
        assert!(success == 0 || success == 1);
        assert_eq!(out.read().wait()[13], if success == 1 { 15 } else { 6 });
        assert_eq!(&out.read().wait()[14..], &[i32::MAX, i32::MIN]);
    }
    Ok(())
}

/// One thread runs the kernel; the threadgroup variant has no device cell parameter.
fn run<T: metal_oxide::GpuAtomic>(
    pipeline: &Pipeline,
    shared: bool,
    cell: &mut metal_oxide::Buffer<T>,
    out: &mut metal_oxide::Buffer<T>,
    value: T,
) -> metal_oxide::Result<()> {
    let config = LaunchConfig::<1>::new(Dim3::x(1));
    // SAFETY: one thread operates on one initialized cell and sixteen separate output elements.
    unsafe {
        if shared {
            Launch::new(Ok(pipeline), config, (out, value), |(out, value)| {
                Ok([Argument::write(out), Argument::value(*value)?])
            })
            .wait()?;
        } else {
            Launch::new(
                Ok(pipeline),
                config,
                (cell, out, value),
                |(cell, out, value)| {
                    Ok([
                        Argument::atomic(cell),
                        Argument::write(out),
                        Argument::value(*value)?,
                    ])
                },
            )
            .wait()?;
        }
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
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<256>::new(Dim3::x(257)),
            (&mut output,),
            |(output,)| Ok([Argument::write(output)]),
        )
    }
    .wait()?;
    assert!(output.read().wait().iter().all(|value| *value == 128));
    Ok(())
}
