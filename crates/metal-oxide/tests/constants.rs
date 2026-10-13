#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    task::{Context, Waker},
};

use metal_oxide::{Argument, Device, Launch, LaunchConfig, Module, Pipeline, sequence};

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn constant_bindings_survive_reuse_and_forgotten_operations() -> metal_oxide::Result<()> {
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
        let config = LaunchConfig::<1>::for_elements(1)?;
        // SAFETY: each one-thread dispatch owns one output, and each initialized
        // constant array covers its count. The scalar arguments have matching layouts.
        let chain = sequence(|batch| unsafe {
            batch.push(Launch::new(
                Ok(pipeline),
                config,
                (&mut first, tag),
                |(first, tag)| {
                    Ok([
                        Argument::write(first),
                        Argument::value([7_u32; N])?,
                        Argument::value(N as u32)?,
                        Argument::value(*tag)?,
                    ])
                },
            ))?;
            batch.push(Launch::new(
                Ok(pipeline),
                config,
                (&mut second, tag + 1),
                |(second, tag)| {
                    Ok([
                        Argument::write(second),
                        Argument::value([11_u32; N])?,
                        Argument::value(N as u32)?,
                        Argument::value(*tag)?,
                    ])
                },
            ))?;
            Ok(())
        });
        let mut future = chain.into_future();
        let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
        std::mem::forget(future);
        assert_eq!(first.read().wait(), [14 + tag]);
        assert_eq!(second.read().wait(), [23 + tag]);
    }
    Ok(())
}
