#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use metal_oxide::{Argument, Device, LaunchConfig, Module, Pipeline};

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn constant_bindings_survive_reuse_and_forgotten_submissions() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let module = Module::from_source(
        &device,
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void constants(device ulong *out [[buffer(0)]],
            constant uint *values [[buffer(1)]], constant uint &count [[buffer(2)]],
            constant ulong &tag [[buffer(3)]]) {
            out[0] = ulong(values[0]) + values[count - 1] + tag;
        }
    "#,
    )?;
    let pipeline = Pipeline::new(&device, &module, "constants")?;
    check::<1>(&device, &pipeline)?;
    check::<1024>(&device, &pipeline)?;
    check::<1025>(&device, &pipeline)
}

fn check<const N: usize>(device: &Device, pipeline: &Pipeline) -> metal_oxide::Result<()> {
    let mut first = device.buffer_zeroed::<u64>(1)?;
    let mut second = device.buffer_zeroed::<u64>(1)?;
    for iteration in 0..3_u64 {
        let tag = (1_u64 << 40) + iteration;
        // SAFETY: each one-thread dispatch owns one output, and each initialized
        // constant array covers its count. The scalar arguments have matching layouts.
        let submission = unsafe {
            device.submit(|batch| {
                batch.launch(
                    pipeline,
                    LaunchConfig::<1>::for_elements(1)?,
                    &[
                        Argument::write(&mut first),
                        Argument::value([7_u32; N])?,
                        Argument::value(N as u32)?,
                        Argument::value(tag)?,
                    ],
                )?;
                batch.launch(
                    pipeline,
                    LaunchConfig::<1>::for_elements(1)?,
                    &[
                        Argument::write(&mut second),
                        Argument::value([11_u32; N])?,
                        Argument::value(N as u32)?,
                        Argument::value(tag + 1)?,
                    ],
                )
            })?
        };
        std::mem::forget(submission);
        assert_eq!(first.as_slice(), [14 + tag]);
        assert_eq!(second.as_slice(), [23 + tag]);
    }
    Ok(())
}
