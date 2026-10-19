#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

use std::{sync::Arc, time::Duration};

use metal_oxide::{
    Argument, Device, Launch, LaunchConfig, Module, Pipeline, ReadArgument, Result, WriteArgument,
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
