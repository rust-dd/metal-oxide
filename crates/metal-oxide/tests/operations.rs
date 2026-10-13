#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{
    future::{Future, IntoFuture},
    pin::{Pin, pin},
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use metal_oxide::{
    Argument, Device, Dim3, Error, Launch, LaunchConfig, Module, Operation, Pipeline, sequence,
};

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

type Arguments<I, O> = (I, O, f32, f32, u32);

/// `output[i] = input[i] * scale + offset` over `n` elements.
fn scale_add<I, O>(
    pipeline: &Pipeline,
    config: LaunchConfig<256>,
    input: I,
    output: O,
    scale: f32,
    offset: f32,
    n: u32,
) -> Launch<Arguments<I, O>, 5>
where
    I: metal_oxide::ReadArgument<f32>,
    O: metal_oxide::WriteArgument<f32>,
{
    // SAFETY: callers pass n-element buffers in distinct allocations; the kernel guards n.
    unsafe {
        Launch::new(
            Ok(pipeline),
            config,
            (input, output, scale, offset, n),
            |(input, output, scale, offset, n)| {
                Ok([
                    Argument::read(input.buffer()),
                    Argument::write(output.buffer_mut()),
                    Argument::value(*scale)?,
                    Argument::value(*offset)?,
                    Argument::value(*n)?,
                ])
            },
        )
    }
}

struct Unpark(std::thread::Thread);
impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn await_on_thread<F: IntoFuture>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut future = pin!(future.into_future());
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Polls an operation once, which encodes and commits it, and returns its future.
fn started<F: IntoFuture<IntoFuture: Unpin>>(operation: F) -> F::IntoFuture {
    let mut future = operation.into_future();
    let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
    future
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn timed_operations_report_gpu_time_and_explicit_upload_preserves_gpu_writes()
-> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let mut input = device.buffer_zeroed::<f32>(257)?;
    let mut output = device.buffer_zeroed::<f32>(257)?;
    let empty = LaunchConfig::<256>::new(Dim3::x(0));
    let (_, time) = scale_add(&pipeline, empty, &input, &mut output, 1.0, 0.0, 0)
        .timed()
        .wait()?;
    assert!(time.is_none());
    for value in [2.0, -5.0] {
        input.write().wait().fill(value);
        input.upload();
        let start = std::time::Instant::now();
        let config = LaunchConfig::<256>::for_elements(257)?;
        let operation = scale_add(&pipeline, config, &input, &mut output, 3.0, 1.0, 257);
        let (_, time) = await_on_thread(operation.timed())?;
        let gpu = time.expect("native completion timestamps");
        assert!(!gpu.is_zero());
        assert!(gpu <= start.elapsed());
        output.upload();
        assert!(output.read().wait().iter().all(|&x| x == value * 3.0 + 1.0));
    }
    Ok(())
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
        let chain = sequence(|batch| {
            let (_, intermediate, ..) = batch.push(scale_add(
                &pipeline,
                config,
                &input,
                &mut intermediate,
                2.0,
                3.0,
                n,
            ))?;
            batch.push(scale_add(
                &pipeline,
                config,
                &*intermediate,
                &mut output,
                5.0,
                7.0,
                n,
            ))?;
            Ok(())
        });
        await_on_thread(chain)?;
        for (i, actual) in output.read().wait().iter().enumerate() {
            assert_eq!(*actual, (values[i] * 2.0 + 3.0) * 5.0 + 7.0, "n={n}, i={i}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn operations_dropped_before_their_first_poll_encode_nothing() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let n = 1_000_003_u32;
    let input = device.buffer_from_slice(&vec![2.0_f32; n as usize])?;
    let mut output = device.buffer_from_slice(&vec![-1.0_f32; n as usize])?;
    let config = LaunchConfig::<256>::for_elements(n)?;
    let operation = scale_add(&pipeline, config, &input, &mut output, 3.0, 1.0, n);
    drop(operation);
    let future = scale_add(&pipeline, config, &input, &mut output, 5.0, 1.0, n).into_future();
    drop(future);
    let mut other = device.buffer_zeroed::<f32>(n as usize)?;
    scale_add(&pipeline, config, &input, &mut other, 1.0, 0.0, n).wait()?;
    assert!(output.read().wait().iter().all(|&value| value == -1.0));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn dropped_and_forgotten_running_operations_synchronize_cpu_access() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let n = 1_000_003_u32;
    let input = device.buffer_from_slice(&vec![2.0_f32; n as usize])?;
    let mut output = device.buffer_zeroed::<f32>(n as usize)?;
    let config = LaunchConfig::<256>::for_elements(n)?;
    for forget in [false, true] {
        let future = started(scale_add(
            &pipeline,
            config,
            &input,
            &mut output,
            3.0,
            1.0,
            n,
        ));
        if forget {
            std::mem::forget(future);
        } else {
            drop(future);
        }
        assert!(output.read().wait().iter().all(|&value| value == 7.0));
        output.write().wait().fill(-1.0);
    }
    drop(started(scale_add(
        &pipeline,
        config,
        &input,
        &mut output,
        2.0,
        5.0,
        n,
    )));
    let mut last = device.buffer_zeroed::<f32>(n as usize)?;
    scale_add(&pipeline, config, &output, &mut last, 3.0, 2.0, n).wait()?;
    assert!(last.read().wait().iter().all(|&value| value == 29.0));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn owned_arguments_come_back_after_the_gpu_work() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let n = 1_000_003_u32;
    let input = Arc::new(device.buffer_from_slice(&vec![4.0_f32; n as usize])?);
    let output = device.buffer_zeroed::<f32>(n as usize)?;
    let config = LaunchConfig::<256>::for_elements(n)?;
    let (input, output, ..) = scale_add(&pipeline, config, input, output, 3.0, 2.0, n).wait()?;
    assert!(output.read().wait().iter().all(|&value| value == 14.0));
    assert_eq!(Arc::strong_count(&input), 1);
    let future = started(scale_add(&pipeline, config, input, output, 1.0, 1.0, n));
    // The GPU still reads and writes buffers whose owners were dropped with the future.
    drop(future);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn an_encoding_error_discards_the_chain_and_leaves_the_context_usable() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let input = device.buffer_from_slice(&[1.0_f32; 32])?;
    let mut output = device.buffer_zeroed::<f32>(32)?;
    let config = LaunchConfig::<256>::new(Dim3::x(1));
    for (offset, reject, expected) in [
        (1.0_f32, true, 0.0),
        (2.0, false, 3.0),
        (3.0, true, 3.0),
        (4.0, false, 5.0),
    ] {
        let result = sequence(|batch| {
            batch.push(scale_add(
                &pipeline,
                config,
                &input,
                &mut output,
                1.0,
                offset,
                32,
            ))?;
            if reject {
                // SAFETY: validation rejects the empty block before anything is encoded.
                batch.push(unsafe {
                    Launch::new(
                        Ok(&pipeline),
                        LaunchConfig::<0>::new(Dim3::x(1)),
                        (),
                        |()| Ok([]),
                    )
                })?;
            }
            Ok(())
        })
        .wait();
        assert_eq!(
            matches!(result, Err(Error::InvalidLaunch(_))),
            reject,
            "{result:?}"
        );
        assert_eq!(output.read().wait(), [expected; 32]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn resources_from_another_queue_context_are_rejected() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let other = Device::system_default()?;
    let foreign = other.buffer_zeroed::<f32>(32)?;
    let mut output = device.buffer_zeroed::<f32>(32)?;
    let config = LaunchConfig::<256>::new(Dim3::x(0));
    let result = scale_add(&pipeline, config, &foreign, &mut output, 1.0, 0.0, 0).wait();
    assert!(matches!(result, Err(Error::DeviceMismatch)));
    let module = Module::from_source(&other, SOURCE)?;
    let other_pipeline = Pipeline::new(&other, &module, "scale_add")?;
    let input = device.buffer_zeroed::<f32>(32)?;
    let result = sequence(|batch| {
        let (input, output, ..) = batch.push(scale_add(
            &pipeline,
            config,
            &input,
            &mut output,
            1.0,
            0.0,
            0,
        ))?;
        batch.push(scale_add(
            &other_pipeline,
            config,
            input,
            output,
            1.0,
            0.0,
            0,
        ))
    })
    .wait();
    assert!(matches!(result, Err(Error::DeviceMismatch)));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn overlapping_operations_survive_cancellation_and_reuse() -> metal_oxide::Result<()> {
    let (device, pipeline) = setup()?;
    let n = 65_537_u32;
    let input = device.buffer_from_slice(&vec![2.0_f32; n as usize])?;
    let mut outputs = (0..32)
        .map(|_| device.buffer_zeroed::<f32>(n as usize))
        .collect::<metal_oxide::Result<Vec<_>>>()?;
    let config = LaunchConfig::<256>::for_elements(n)?;
    for round in 0..3 {
        let mut pending = Vec::new();
        for (index, output) in outputs.iter_mut().enumerate() {
            let offset = (round * 32 + index) as f32;
            let future = started(scale_add(&pipeline, config, &input, output, 3.0, offset, n));
            match index % 3 {
                0 => pending.push(future),
                1 => drop(future),
                _ => std::mem::forget(future),
            }
        }
        for future in pending.into_iter().rev() {
            await_on_thread(future)?;
        }
        for (index, output) in outputs.iter().enumerate() {
            assert!(
                output
                    .read()
                    .wait()
                    .iter()
                    .all(|&value| value == (6 + round * 32 + index) as f32)
            );
        }
    }
    Ok(())
}
