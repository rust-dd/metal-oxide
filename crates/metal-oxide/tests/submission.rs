#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use metal_oxide::{Argument, Device, Dim3, Error, LaunchConfig, Module, Pipeline};

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

fn setup() -> metal_oxide::Result<(Device, Pipeline)> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "scale_add")?;
    Ok((device, pipeline))
}

struct Unpark(std::thread::Thread);
impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn await_on_thread<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park(),
        }
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn chained_kernels_publish_intermediate_writes_to_the_next_dispatch() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
        let values = (0..n).map(|i| (i % 1024) as f32).collect::<Vec<_>>();
        let input = device.buffer_from_slice(&values)?;
        let mut intermediate = device.buffer_zeroed::<f32>(n as usize)?;
        let mut output = device.buffer_zeroed::<f32>(n as usize)?;
        let config = LaunchConfig::<256>::for_elements(n)?;
        // SAFETY: each pass writes a distinct allocation, guards n, and consumes the prior pass in order.
        let submission = unsafe {
            device.submit(|batch| {
                batch.launch(
                    &pipeline,
                    config,
                    &[
                        Argument::read(&input),
                        Argument::write(&mut intermediate),
                        Argument::value::<f32>(2.0).unwrap(),
                        Argument::value::<f32>(3.0).unwrap(),
                        Argument::value::<u32>(n).unwrap(),
                    ],
                )?;
                batch.launch(
                    &pipeline,
                    config,
                    &[
                        Argument::read(&intermediate),
                        Argument::write(&mut output),
                        Argument::value::<f32>(5.0).unwrap(),
                        Argument::value::<f32>(7.0).unwrap(),
                        Argument::value::<u32>(n).unwrap(),
                    ],
                )
            })
        }?;
        await_on_thread(submission)?;
        for (i, actual) in output.as_slice().iter().enumerate() {
            assert_eq!(*actual, (values[i] * 2.0 + 3.0) * 5.0 + 7.0, "n={n}, i={i}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn cancelled_and_forgotten_submissions_synchronize_cpu_access() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let n = 1_000_003_u32;
    let input = device.buffer_from_slice(&vec![2.0_f32; n as usize])?;
    let mut output = device.buffer_zeroed::<f32>(n as usize)?;
    let config = LaunchConfig::<256>::for_elements(n)?;
    for forget in [false, true] {
        // SAFETY: independent allocations cover n elements and the kernel guards the padded grid.
        let submission = unsafe {
            device.submit(|batch| {
                batch.launch(
                    &pipeline,
                    config,
                    &[
                        Argument::read(&input),
                        Argument::write(&mut output),
                        Argument::value::<f32>(3.0).unwrap(),
                        Argument::value::<f32>(1.0).unwrap(),
                        Argument::value::<u32>(n).unwrap(),
                    ],
                )
            })
        }?;
        if forget {
            std::mem::forget(submission);
        } else {
            drop(submission);
        }
        assert!(output.as_slice().iter().all(|&value| value == 7.0));
        output.as_mut_slice().fill(-1.0);
    }
    // SAFETY: two ordered submissions access the same buffer after the first future is dropped.
    unsafe {
        drop(device.submit(|batch| {
            batch.launch(
                &pipeline,
                config,
                &[
                    Argument::read(&input),
                    Argument::write(&mut output),
                    Argument::value::<f32>(2.0).unwrap(),
                    Argument::value::<f32>(5.0).unwrap(),
                    Argument::value::<u32>(n).unwrap(),
                ],
            )
        })?);
        let mut final_output = device.buffer_zeroed::<f32>(n as usize)?;
        device
            .submit(|batch| {
                batch.launch(
                    &pipeline,
                    config,
                    &[
                        Argument::read(&output),
                        Argument::write(&mut final_output),
                        Argument::value::<f32>(3.0).unwrap(),
                        Argument::value::<f32>(2.0).unwrap(),
                        Argument::value::<u32>(n).unwrap(),
                    ],
                )
            })?
            .wait()?;
        assert!(final_output.as_slice().iter().all(|&value| value == 29.0));
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn submission_retains_resources_owned_by_the_encoding_closure() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let n = 1_000_003_u32;
    let input = device.buffer_from_slice(&vec![4.0_f32; n as usize])?;
    let mut output = device.buffer_zeroed::<f32>(n as usize)?;
    let output_ref = &mut output;
    // SAFETY: the closure's owned input and pipeline are retained by the command until completion.
    unsafe {
        device.submit(move |batch| {
            batch.launch(
                &pipeline,
                LaunchConfig::<256>::for_elements(n)?,
                &[
                    Argument::read(&input),
                    Argument::write(output_ref),
                    Argument::value::<f32>(3.0).unwrap(),
                    Argument::value::<f32>(2.0).unwrap(),
                    Argument::value::<u32>(n).unwrap(),
                ],
            )
        })
    }?
    .wait()?;
    assert!(output.as_slice().iter().all(|&value| value == 14.0));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn encoding_error_discards_the_whole_batch() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let input = device.buffer_from_slice(&[1.0_f32; 32])?;
    let mut output = device.buffer_zeroed::<f32>(32)?;
    // SAFETY: the valid first dispatch has distinct sized resources; the second is rejected before encoding.
    let result = unsafe {
        device.submit(|batch| {
            batch.launch(
                &pipeline,
                LaunchConfig::<32>::new(Dim3::x(1)),
                &[
                    Argument::read(&input),
                    Argument::write(&mut output),
                    Argument::value::<f32>(1.0).unwrap(),
                    Argument::value::<f32>(1.0).unwrap(),
                    Argument::value::<u32>(32).unwrap(),
                ],
            )?;
            batch.launch(&pipeline, LaunchConfig::<0>::new(Dim3::x(1)), &[])
        })
    };
    assert!(matches!(result, Err(Error::InvalidLaunch(_))));
    drop(result);
    assert_eq!(output.as_slice(), [0.0_f32; 32]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn resources_from_another_queue_context_are_rejected() -> metal_oxide::Result<()> {
    let (_device, pipeline) = setup()?;
    let other = Device::system_default()?;
    // SAFETY: no kernel runs because the pipeline belongs to another context.
    let result = unsafe { other.launch(&pipeline, LaunchConfig::<32>::new(Dim3::x(0)), &[]) };
    assert!(matches!(result, Err(Error::DeviceMismatch)));
    Ok(())
}
