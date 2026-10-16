#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Waker},
    time::{Duration, Instant},
};

use metal_oxide::{
    Argument, Buffer, Device, Launch, LaunchConfig, Module, Operation, Pipeline, ReadArgument,
    Result, WriteArgument,
};
use tokio::runtime::Builder;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void scale_add(device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]], constant float& scale [[buffer(2)]],
    constant float& offset [[buffer(3)]], constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) output[i] = input[i] * scale + offset;
}
kernel void spin(device uint* out [[buffer(0)]], constant uint& rounds [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    uint x = i;
    for (uint round = 0; round < rounds; ++round) {
        x = x * 1664525u + 1013904223u;
    }
    out[i] = x;
}
"#;

fn setup(kernel: &str) -> Result<(Device, Pipeline)> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, kernel)?;
    Ok((device, pipeline))
}

/// `output = input * scale + offset` over n elements.
///
/// # Safety
///
/// Both buffers must hold at least n elements.
unsafe fn scale_add<I: ReadArgument<f32>, O: WriteArgument<f32>>(
    pipeline: &Pipeline,
    input: I,
    output: O,
    scale: f32,
    n: u32,
) -> Launch<(I, O, f32, u32), 5> {
    let config = LaunchConfig::<256>::for_elements(n).unwrap();
    // SAFETY: forwarded from the caller; the kernel guards the padded grid.
    unsafe {
        Launch::new(
            Ok(pipeline),
            config,
            (input, output, scale, n),
            |(input, output, scale, n)| {
                Ok([
                    Argument::read(input.buffer()),
                    Argument::write(output.buffer_mut()),
                    Argument::value(*scale)?,
                    Argument::value(1.0_f32)?,
                    Argument::value(*n)?,
                ])
            },
        )
    }
}

/// One lane per element iterates an LCG `rounds` times, which keeps the GPU busy long enough
/// to observe waiting.
fn spin<'a>(
    pipeline: &Pipeline,
    out: &'a mut Buffer<u32>,
    rounds: u32,
) -> Launch<(&'a mut Buffer<u32>, u32), 2> {
    let config = LaunchConfig::<256>::for_elements(out.len() as u32).unwrap();
    // SAFETY: one lane writes each element of `out`.
    unsafe {
        Launch::new(Ok(pipeline), config, (out, rounds), |(out, rounds)| {
            Ok([Argument::write(out), Argument::value(*rounds)?])
        })
    }
}

fn started<F: IntoFuture<IntoFuture: Unpin>>(operation: F) -> F::IntoFuture {
    let mut future = operation.into_future();
    let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
    future
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn spawned_tasks_run_owned_operations_on_a_multi_thread_runtime() -> Result<()> {
    let (device, pipeline) = setup("scale_add")?;
    let runtime = Builder::new_multi_thread().worker_threads(4).build()?;
    let n = 65_537_u32;
    runtime.block_on(async {
        let input = Arc::new(device.buffer_from_slice(&vec![3.0_f32; n as usize])?);
        let tasks = (0..32_u32)
            .map(|task| {
                let (device, pipeline, input) = (device.clone(), pipeline.clone(), input.clone());
                tokio::spawn(async move {
                    let output = device.buffer_zeroed::<f32>(n as usize)?;
                    // SAFETY: both buffers hold n elements.
                    let operation = unsafe { scale_add(&pipeline, input, output, task as f32, n) };
                    let (input, output, ..) = operation.await?;
                    let expected = 3.0 * task as f32 + 1.0;
                    assert!(output.read().await.iter().all(|&value| value == expected));
                    assert!(input.read().await.iter().all(|&value| value == 3.0));
                    Ok::<_, metal_oxide::Error>(())
                })
            })
            .collect::<Vec<_>>();
        for task in tasks {
            task.await.unwrap()?;
        }
        assert_eq!(Arc::strong_count(&input), 1);
        Ok(())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn ten_thousand_small_operations_complete_concurrently() -> Result<()> {
    let (device, pipeline) = setup("scale_add")?;
    let runtime = Builder::new_multi_thread().worker_threads(8).build()?;
    let input = Arc::new(device.buffer_from_slice(&[2.0_f32])?);
    runtime.block_on(async {
        let tasks = (0..10_000_u32)
            .map(|task| {
                let (device, pipeline, input) = (device.clone(), pipeline.clone(), input.clone());
                tokio::spawn(async move {
                    let output = device.buffer_zeroed::<f32>(1)?;
                    // SAFETY: both buffers hold one element.
                    let (_, output, ..) =
                        unsafe { scale_add(&pipeline, input, output, task as f32, 1) }.await?;
                    assert_eq!(output.read().await, [2.0 * task as f32 + 1.0]);
                    Ok::<_, metal_oxide::Error>(())
                })
            })
            .collect::<Vec<_>>();
        for task in tasks {
            task.await.unwrap()?;
        }
        Ok(())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn operations_dropped_by_a_timeout_keep_readback_ordered() -> Result<()> {
    let (device, pipeline) = setup("scale_add")?;
    let runtime = Builder::new_current_thread().enable_time().build()?;
    let n = 4_000_037_u32;
    runtime.block_on(async {
        let input = device.buffer_from_slice(&vec![5.0_f32; n as usize])?;
        let mut output = device.buffer_zeroed::<f32>(n as usize)?;
        for round in 0..4 {
            // SAFETY: both buffers hold n elements.
            let operation = unsafe { scale_add(&pipeline, &input, &mut output, round as f32, n) };
            let timed_out = tokio::time::timeout(Duration::from_nanos(1), operation).await;
            if let Ok(result) = timed_out {
                result?;
            }
            let expected = 5.0 * round as f32 + 1.0;
            assert!(output.read().await.iter().all(|&value| value == expected));
        }
        Ok(())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn waiting_for_the_gpu_never_blocks_a_current_thread_runtime() -> Result<()> {
    let (device, pipeline) = setup("spin")?;
    let runtime = Builder::new_current_thread().enable_time().build()?;
    let mut out = device.buffer_zeroed::<u32>(32_768)?;
    // Calibrate the kernel to run for a clearly observable time on this GPU.
    let mut rounds = 1 << 12;
    loop {
        let start = Instant::now();
        spin(&pipeline, &mut out, rounds).wait()?;
        if start.elapsed() >= Duration::from_millis(60) || rounds >= 1 << 28 {
            break;
        }
        rounds *= 2;
    }
    runtime.block_on(async {
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticker = tokio::spawn({
            let ticks = Arc::clone(&ticks);
            async move {
                loop {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    ticks.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        tokio::task::yield_now().await;
        let before = ticks.load(Ordering::Relaxed);
        spin(&pipeline, &mut out, rounds).await?;
        let during_operation = ticks.load(Ordering::Relaxed) - before;
        drop(started(spin(&pipeline, &mut out, rounds)));
        let before = ticks.load(Ordering::Relaxed);
        let values = out.read().await;
        let during_readback = ticks.load(Ordering::Relaxed) - before;
        ticker.abort();
        assert_eq!(values.len(), 32_768);
        // A blocking wait lets the ticker run at most once; virtual GPUs coarsen its period.
        assert!(
            during_operation >= 3,
            "{during_operation} ticks while awaiting"
        );
        assert!(
            during_readback >= 3,
            "{during_readback} ticks during readback"
        );
        Ok(())
    })
}
