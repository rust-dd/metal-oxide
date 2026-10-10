#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Device, Dim3, Error, LaunchConfig, Module, Pipeline};

#[test]
fn threadgroup_lengths_check_multiplication_and_native_rounding() {
    assert!(Argument::threadgroup::<u64>(0).is_ok());
    assert!(Argument::threadgroup::<u8>(1).is_ok());
    assert!(matches!(
        Argument::threadgroup::<u64>(usize::MAX),
        Err(Error::LengthOverflow)
    ));
    assert!(matches!(
        Argument::threadgroup::<u8>(usize::MAX),
        Err(Error::LengthOverflow)
    ));
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn threadgroup_memory_limits_include_static_storage_and_empty_grids() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(
        &device,
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void shared_total(threadgroup uint* a [[threadgroup(0)]], device uint* out [[buffer(0)]], uint i [[thread_position_in_threadgroup]]) {
            threadgroup uint b[32];
            b[i] = i; a[i] = i + 2;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out[i] = b[31-i] + a[i];
        }
        kernel void two(threadgroup uint* a [[threadgroup(0)]], threadgroup uint* b [[threadgroup(1)]], device uint* out [[buffer(0)]], uint i [[thread_position_in_threadgroup]]) {
            a[i] = i; b[i] = 1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out[i] = a[31-i] + b[i];
        }
        kernel void guarded(threadgroup uchar* a [[threadgroup(0)]], device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_threadgroup]]) {
            if (i < n) a[i] = uchar(i);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i == 0) out[0] = n == 0 ? 42 : uint(a[n-1]);
        }
    "#,
    )?;
    let pipeline = Pipeline::new(&device, &module, "shared_total")?;
    let mut output = device.buffer_zeroed::<u32>(32)?;
    let maximum = device.max_threadgroup_memory_per_block();
    for grid in [0, 1] {
        // SAFETY: validation must reject combined static/dynamic capacity before any dispatch.
        let result = unsafe {
            device.launch(
                &pipeline,
                LaunchConfig::<32>::new(Dim3::x(grid)),
                &[
                    Argument::threadgroup::<u32>(maximum / 4)?,
                    Argument::write(&mut output),
                ],
            )
        };
        assert!(matches!(result, Err(Error::InvalidLaunch(_))));
    }
    // SAFETY: every thread owns one initialized scratch element in each 32-element array.
    unsafe {
        device.launch(
            &pipeline,
            LaunchConfig::<32>::new(Dim3::x(1)),
            &[
                Argument::threadgroup::<u32>(32)?,
                Argument::write(&mut output),
            ],
        )?;
    }
    assert_eq!(output.as_slice(), &[33; 32]);

    let two = Pipeline::new(&device, &module, "two")?;
    let guarded = Pipeline::new(&device, &module, "guarded")?;
    // SAFETY: every accessed cell is initialized within its allocation; each
    // dispatch completes before the next uses output. Scratch is block-local.
    unsafe {
        device
            .submit(|batch| {
                batch.launch(
                    &two,
                    LaunchConfig::<32>::new(Dim3::x(1)),
                    &[
                        Argument::threadgroup::<u32>(maximum / 8)?,
                        Argument::threadgroup::<u32>(maximum / 8)?,
                        Argument::write(&mut output),
                    ],
                )?;
                batch.launch(
                    &pipeline,
                    LaunchConfig::<32>::new(Dim3::x(1)),
                    &[
                        Argument::threadgroup::<u32>((maximum - 128) / 4)?,
                        Argument::write(&mut output),
                    ],
                )
            })?
            .wait()?;
    }
    assert_eq!(output.as_slice(), &[33; 32]);
    // SAFETY: n is at most one block and the zero case accesses no scratch elements.
    unsafe {
        device
            .submit(|batch| {
                for n in [32_u32, 1, 17, 0] {
                    batch.launch(
                        &guarded,
                        LaunchConfig::<32>::new(Dim3::x(1)),
                        &[
                            Argument::threadgroup::<u8>(n as usize)?,
                            Argument::write(&mut output),
                            Argument::value(n)?,
                        ],
                    )?;
                }
                Ok(())
            })?
            .wait()?;
    }
    assert_eq!(output.as_slice()[0], 42);
    Ok(())
}
