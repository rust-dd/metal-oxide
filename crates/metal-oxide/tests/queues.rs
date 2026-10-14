#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    task::{Context, Waker},
};

use metal_oxide::{
    Argument, Buffer, Device, DynamicLaunchConfig, Launch, Module, Operation, Pipeline,
};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void late_fill(device float* data [[buffer(0)]], constant uint& iters [[buffer(1)]],
    constant float& value [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float delay = float(i);
    for (uint k = 0; k < iters; ++k) { delay = fma(delay, 0.999f, 0.001f); }
    data[i] = delay == -1.0f ? 0.0f : value;
}
kernel void late_copy(device const float* source [[buffer(0)]], device float* target [[buffer(1)]],
    constant uint& iters [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float delay = float(i);
    for (uint k = 0; k < iters; ++k) { delay = fma(delay, 0.999f, 0.001f); }
    target[i] = delay == -1.0f ? 0.0f : source[i];
}
kernel void add(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = a[i] + b[i];
}
"#;

/// Eight blocks leave most of the GPU idle, so work on other queues can run beside them.
const ELEMENTS: usize = 8 * 256;
const SLOW: u32 = 200_000;

fn config() -> DynamicLaunchConfig {
    DynamicLaunchConfig::for_elements(ELEMENTS as u32, 256).unwrap()
}

fn pipelines(device: &Device) -> metal_oxide::Result<[Pipeline; 3]> {
    let module = Module::from_source(device, SOURCE)?;
    Ok([
        Pipeline::new(device, &module, "late_fill")?,
        Pipeline::new(device, &module, "late_copy")?,
        Pipeline::new(device, &module, "add")?,
    ])
}

/// Commits an operation and drops its future: the GPU keeps running it, and the runtime still
/// orders later work on its buffers after it.
fn detach<O: Operation + IntoFuture<IntoFuture: Unpin>>(operation: O) {
    let mut future = operation.into_future();
    let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
}

/// Writes `value` to every element after `delay` iterations.
fn fill<'a>(
    pipeline: &'a Pipeline,
    data: &'a mut Buffer<f32>,
    value: f32,
    delay: u32,
) -> Launch<(&'a mut Buffer<f32>, f32, u32), 3> {
    // SAFETY: each thread writes its own element of an ELEMENTS-element buffer.
    unsafe {
        Launch::new(
            Ok(pipeline),
            config(),
            (data, value, delay),
            |(data, value, delay)| {
                Ok([
                    Argument::write(data),
                    Argument::value(*delay)?,
                    Argument::value(*value)?,
                ])
            },
        )
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_read_follows_writes_still_running_on_other_queues() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let [late_fill, _, add] = pipelines(&device)?;
    for round in 0..4 {
        let mut a = device.buffer_zeroed::<f32>(ELEMENTS)?;
        let mut b = device.buffer_zeroed::<f32>(ELEMENTS)?;
        let mut sum = device.buffer_zeroed::<f32>(ELEMENTS)?;
        detach(fill(&late_fill, &mut a, 5.0, SLOW));
        detach(fill(&late_fill, &mut b, 7.0 + round as f32, SLOW));
        // SAFETY: each thread writes its own element; the inputs cover the grid.
        unsafe {
            Launch::new(Ok(&add), config(), (&a, &b, &mut sum), |(a, b, sum)| {
                Ok([Argument::read(a), Argument::read(b), Argument::write(sum)])
            })
        }
        .wait()?;
        assert!(
            sum.read()
                .wait()
                .iter()
                .all(|value| *value == 12.0 + round as f32)
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_write_follows_reads_still_running() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let [late_fill, copy, _] = pipelines(&device)?;
    let mut source = device.buffer_from_slice(&[3.0_f32; ELEMENTS])?;
    let mut target = device.buffer_zeroed::<f32>(ELEMENTS)?;
    // SAFETY: each thread copies its own element; both buffers cover the grid.
    detach(unsafe {
        Launch::new(
            Ok(&copy),
            config(),
            (&source, &mut target),
            |(source, target)| {
                Ok([
                    Argument::read(source),
                    Argument::write(target),
                    Argument::value(SLOW)?,
                ])
            },
        )
    });
    // Unordered, the immediate write would land before the copy's late read.
    fill(&late_fill, &mut source, 9.0, 0).wait()?;
    assert!(target.read().wait().iter().all(|value| *value == 3.0));
    assert!(source.read().wait().iter().all(|value| *value == 9.0));
    Ok(())
}
