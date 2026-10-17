#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    pin::Pin,
    task::{Context, Waker},
};

use metal_oxide::{
    Argument, BatchOperation, Buffer, Device, Dim3, Error, GpuValue, Launch, LaunchConfig, Layout,
    Module, Operation, Pipeline, sequence,
};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Value(u32);

impl GpuValue for Value {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;
    fn layout() -> metal_oxide::Result<Layout> {
        u32::layout()
    }
    fn zeroed() -> Self {
        Self(0)
    }
    fn encode(self, bytes: &mut [u8]) {
        self.0.encode(bytes);
    }
    fn decode(bytes: &[u8]) -> Self {
        Self(u32::decode(bytes))
    }
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void set(device uint* value [[buffer(0)]]) { value[0] = 42; }
kernel void copy_value(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]]) {
    output[0] = input[0];
}
"#;

fn setup() -> metal_oxide::Result<(Device, Pipeline, Pipeline)> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    Ok((
        device.clone(),
        Pipeline::new(&device, &module, "set")?,
        Pipeline::new(&device, &module, "copy_value")?,
    ))
}

fn set<'a, T: GpuValue>(
    pipeline: &Pipeline,
    buffer: &'a mut Buffer<T>,
) -> Launch<&'a mut Buffer<T>, 1> {
    // SAFETY: tests pass one-element buffers with the canonical u32 layout and one thread.
    unsafe {
        Launch::new(
            Ok(pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            buffer,
            |buffer| Ok([Argument::write(buffer)]),
        )
    }
}

fn copy<'a, T: GpuValue>(
    pipeline: &Pipeline,
    input: &'a Buffer<T>,
    output: &'a mut Buffer<T>,
) -> Launch<(&'a Buffer<T>, &'a mut Buffer<T>), 2> {
    // SAFETY: both buffers hold one canonical u32 and the single thread accesses element zero.
    unsafe {
        Launch::new(
            Ok(pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            (input, output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
}

fn check_cpu_access<T: GpuValue + std::fmt::Debug + PartialEq>(
    device: &Device,
    pipeline: &Pipeline,
    expected: T,
) -> metal_oxide::Result<()> {
    let mut buffer = device.buffer_zeroed::<T>(1)?;
    sequence(|batch| {
        let buffer = batch.push(set(pipeline, &mut buffer))?;
        assert!(catch_unwind(AssertUnwindSafe(|| buffer.read().wait())).is_err());
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = buffer.write().wait();
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut read = buffer.read();
                let _ = Pin::new(&mut read).poll(&mut Context::from_waker(Waker::noop()));
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut write = buffer.write();
                let _ = Pin::new(&mut write).poll(&mut Context::from_waker(Waker::noop()));
            }))
            .is_err()
        );
        assert!(catch_unwind(AssertUnwindSafe(|| buffer.upload())).is_err());
        Ok(())
    })
    .wait()?;
    assert_eq!(buffer.read().wait(), [expected]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn recorded_writes_exclude_primitive_and_codec_cpu_access() -> metal_oxide::Result<()> {
    let (device, pipeline, _) = setup()?;
    check_cpu_access(&device, &pipeline, 42_u32)?;
    check_cpu_access(&device, &pipeline, Value(42))
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn aborted_and_unwound_recordings_release_their_reservations() -> metal_oxide::Result<()> {
    let (device, pipeline, _) = setup()?;
    let mut buffer = device.buffer_from_slice(&[Value(7)])?;
    let rejected = sequence(|batch| {
        batch.push(set(&pipeline, &mut buffer))?;
        Err::<(), _>(Error::InvalidLaunch("abort recording"))
    })
    .wait();
    assert!(matches!(
        rejected,
        Err(Error::InvalidLaunch("abort recording"))
    ));
    assert_eq!(buffer.read().wait(), [Value(7)]);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            sequence(|batch| -> metal_oxide::Result<()> {
                batch.push(set(&pipeline, &mut buffer))?;
                panic!("abort recording");
            })
            .wait()
        }))
        .is_err()
    );
    assert_eq!(buffer.read().wait(), [Value(7)]);
    set(&pipeline, &mut buffer).wait()?;
    assert_eq!(buffer.read().wait(), [Value(42)]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn nested_conflicts_are_rejected_and_same_batch_chains_succeed() -> metal_oxide::Result<()> {
    let (device, pipeline, copy_pipeline) = setup()?;
    let mut buffer = device.buffer_zeroed::<Value>(1)?;
    let mut output = device.buffer_zeroed::<Value>(1)?;
    sequence(|batch| {
        let buffer = batch.push(set(&pipeline, &mut buffer))?;
        let nested = set(&pipeline, buffer).wait();
        assert!(matches!(nested, Err(Error::InvalidBuffer(_))));
        let nested = copy(&copy_pipeline, buffer, &mut output).wait();
        assert!(matches!(nested, Err(Error::InvalidBuffer(_))));
        batch.push(copy(&copy_pipeline, buffer, &mut output))?;
        batch.push(set(&pipeline, buffer))?;
        Ok(())
    })
    .wait()?;
    assert_eq!(buffer.read().wait(), [Value(42)]);
    assert_eq!(output.read().wait(), [Value(42)]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn read_only_recordings_can_overlap_and_preserve_existing_cpu_views() -> metal_oxide::Result<()> {
    let (device, _, pipeline) = setup()?;
    let input = device.buffer_from_slice(&[Value(9)])?;
    let values = input.read().wait();
    let mut first = device.buffer_zeroed::<Value>(1)?;
    let mut second = device.buffer_zeroed::<Value>(1)?;
    sequence(|batch| {
        batch.push(copy(&pipeline, &input, &mut first))?;
        assert_eq!(input.read().wait(), values);
        copy(&pipeline, &input, &mut second).wait()?;
        assert_eq!(values, [Value(9)]);
        Ok(())
    })
    .wait()?;
    assert_eq!(values, [Value(9)]);
    assert_eq!(first.read().wait(), values);
    assert_eq!(second.read().wait(), values);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn recorded_reads_allow_cpu_reads_and_exclude_cpu_writes() -> metal_oxide::Result<()> {
    let (device, _, pipeline) = setup()?;
    let mut input = device.buffer_from_slice(&[Value(9)])?;
    let mut output = device.buffer_zeroed::<Value>(1)?;
    sequence(|batch| {
        batch.push(copy(&pipeline, &input, &mut output))?;
        assert_eq!(input.read().wait(), [Value(9)]);
        let mut read = input.read();
        assert_eq!(
            Pin::new(&mut read).poll(&mut Context::from_waker(Waker::noop())),
            std::task::Poll::Ready([Value(9)].as_slice())
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = input.write().wait();
            }))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let mut write = input.write();
                let _ = Pin::new(&mut write).poll(&mut Context::from_waker(Waker::noop()));
            }))
            .is_err()
        );
        assert!(catch_unwind(AssertUnwindSafe(|| input.upload())).is_err());
        Ok(())
    })
    .wait()?;
    assert_eq!(input.read().wait(), [Value(9)]);
    assert_eq!(output.read().wait(), [Value(9)]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn raw_encode_reserves_and_empty_grids_do_not() -> metal_oxide::Result<()> {
    let (device, pipeline, _) = setup()?;
    let mut buffer = device.buffer_from_slice(&[Value(7)])?;
    sequence(|batch| {
        // SAFETY: the empty launch dispatches nothing, but its buffer is valid for this kernel.
        let buffer = unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(0)),
                &mut buffer,
                |buffer| Ok([Argument::write(buffer)]),
            )
        }
        .encode(batch)?;
        assert_eq!(buffer.read().wait(), [Value(7)]);
        let buffer = set(&pipeline, buffer).encode(batch)?;
        assert!(catch_unwind(AssertUnwindSafe(|| buffer.read().wait())).is_err());
        Ok(())
    })
    .wait()?;
    assert_eq!(buffer.read().wait(), [Value(42)]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn owned_recorded_buffers_cannot_be_exported() -> metal_oxide::Result<()> {
    let (device, pipeline, _) = setup()?;
    let buffer = device.buffer_from_slice(&[Value(7)])?;
    sequence(|batch| {
        // SAFETY: the owned buffer holds one canonical u32 and exactly one thread writes it.
        let buffer = batch.push(unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<1>::new(Dim3::x(1)),
                buffer,
                |buffer| Ok([Argument::write(buffer)]),
            )
        })?;
        assert!(catch_unwind(AssertUnwindSafe(|| buffer.into_native())).is_err());
        Ok(())
    })
    .wait()
}
