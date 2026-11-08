#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use metal_oxide::{
    Argument, Device, Dim3, Error, Launch, LaunchConfig, Module, Operation, Pipeline, ReadArgument,
    Result, WriteArgument, sequence,
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

fn setup() -> Result<(Device, Pipeline)> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "scale_add")?;
    Ok((device, pipeline))
}

/// `output = input * scale + offset` over the elements of `input`.
fn scale_add<I: ReadArgument<f32>, O: WriteArgument<f32>>(
    pipeline: &Pipeline,
    input: I,
    output: O,
    scale: f32,
    offset: f32,
) -> Launch<(I, O, f32, f32, u32), 5> {
    let n = input.view().len() as u32;
    let config = LaunchConfig::<256>::for_elements(n).unwrap();
    // SAFETY: callers pass equally long buffers in distinct allocations; the kernel guards n.
    unsafe {
        Launch::new(
            Ok(pipeline),
            config,
            (input, output, scale, offset, n),
            |(input, output, scale, offset, n)| {
                Ok([
                    Argument::read(input),
                    Argument::write(output),
                    Argument::value(*scale)?,
                    Argument::value(*offset)?,
                    Argument::value(*n)?,
                ])
            },
        )
    }
}

/// Fails validation before encoding; on success it would return `arguments`.
fn rejected<A>(pipeline: &Pipeline, arguments: A) -> Launch<A, 0> {
    // SAFETY: validation rejects the empty block before anything is encoded.
    unsafe {
        Launch::new(
            Ok(pipeline),
            LaunchConfig::<0>::new(Dim3::x(1)),
            arguments,
            |_| Ok([]),
        )
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn borrowed_launches_feed_readback_and_host_processing() -> Result<()> {
    let (device, pipeline) = setup()?;
    for n in [0, 257, 1_000_003] {
        let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
        let mut middle = device.buffer_zeroed::<f32>(n)?;
        let mut output = device.buffer_from_slice(&vec![-1.0_f32; n])?;
        let fifteens = scale_add(&pipeline, &input, &mut middle, 2.0, 1.0)
            .then(|(_, middle, ..)| scale_add(&pipeline, &*middle, &mut output, 3.0, 0.0))
            .then(|(_, output, ..)| async move { Ok(output.read().await.to_vec()) })
            .map(|values| values.iter().filter(|&&value| value == 15.0).count())
            .wait()?;
        assert_eq!(fifteens, n);
        assert!(middle.read().wait().iter().all(|&value| value == 5.0));
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn owned_chains_run_as_spawned_tasks() -> Result<()> {
    let (device, pipeline) = setup()?;
    let runtime = Builder::new_multi_thread().worker_threads(4).build()?;
    let n = 65_537;
    runtime.block_on(async {
        let mut tasks = Vec::new();
        for task in 0..16_u8 {
            let input = device.buffer_from_slice(&vec![f32::from(task); n])?;
            let middle = device.buffer_zeroed::<f32>(n)?;
            let output = device.buffer_zeroed::<f32>(n)?;
            let next = pipeline.clone();
            let chain = scale_add(&pipeline, input, middle, 2.0, 1.0)
                .then(move |(_, middle, ..)| scale_add(&next, middle, output, 3.0, 0.0))
                .then(|(_, output, ..)| async move { Ok(output.read().await.to_vec()) });
            tasks.push(tokio::spawn(chain));
        }
        for (task, handle) in tasks.into_iter().enumerate() {
            let expected = (task as f32 * 2.0 + 1.0) * 3.0;
            let values = handle.await.unwrap()?;
            assert_eq!(values.len(), n);
            assert!(values.iter().all(|&value| value == expected), "task {task}");
        }
        Ok(())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_failed_stage_returns_its_error_and_starts_no_later_stage() -> Result<()> {
    let (device, pipeline) = setup()?;
    let n = 1_000_003;
    let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
    let mut middle = device.buffer_zeroed::<f32>(n)?;
    let mut output = device.buffer_from_slice(&vec![-1.0_f32; n])?;
    let later = AtomicUsize::new(0);
    let result = scale_add(&pipeline, &input, &mut middle, 2.0, 1.0)
        .then(|(_, middle, ..)| rejected(&pipeline, middle))
        .then(|middle| {
            later.fetch_add(1, Ordering::SeqCst);
            scale_add(&pipeline, &*middle, &mut output, 3.0, 0.0)
        })
        .map(|_| later.fetch_add(1, Ordering::SeqCst))
        .wait();
    assert!(matches!(result, Err(Error::InvalidLaunch(_))), "{result:?}");
    assert_eq!(later.load(Ordering::SeqCst), 0);
    assert!(middle.read().wait().iter().all(|&value| value == 5.0));
    assert!(output.read().wait().iter().all(|&value| value == -1.0));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_shared_launch_runs_once_for_concurrent_readback_and_dependent_launches() -> Result<()> {
    let (device, pipeline) = setup()?;
    let n = 1_000_003;
    let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
    let mut middle = device.buffer_zeroed::<f32>(n)?;
    let mut tripled = device.buffer_from_slice(&vec![-1.0_f32; n])?;
    let (pipeline, tripled_view) = (&pipeline, &mut tripled);
    let encodes = AtomicUsize::new(0);
    let produced = sequence(|batch| {
        encodes.fetch_add(1, Ordering::SeqCst);
        batch.push(scale_add(pipeline, &input, &mut middle, 2.0, 1.0))
    })
    .shared();
    let readback = produced
        .clone()
        .then(|arguments| async move { Ok(arguments.1.read().await.to_vec()) });
    let dependent = produced.clone().then(|arguments| async move {
        scale_add(pipeline, &*arguments.1, tripled_view, 3.0, 0.0)
            .await
            .map(drop)
            .map_err(Arc::new)
    });
    let runtime = Builder::new_current_thread().build()?;
    let (produced, readback, dependent) =
        runtime.block_on(async { tokio::join!(produced, readback, dependent) });
    let (produced, readback) = (produced.unwrap(), readback.unwrap());
    dependent.unwrap();
    assert_eq!(encodes.load(Ordering::SeqCst), 1);
    assert_eq!(Arc::strong_count(&produced), 1);
    assert!(readback.iter().all(|&value| value == 5.0));
    drop(produced);
    assert!(middle.read().wait().iter().all(|&value| value == 5.0));
    assert!(tripled.read().wait().iter().all(|&value| value == 15.0));
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_shared_failure_reaches_every_consumer_and_starts_no_dependent() -> Result<()> {
    let (device, pipeline) = setup()?;
    let mut middle = device.buffer_zeroed::<f32>(257)?;
    let mut output = device.buffer_from_slice(&vec![-1.0_f32; 257])?;
    let later = AtomicUsize::new(0);
    let (pipeline, output_view) = (&pipeline, &mut output);
    let failed = rejected(pipeline, &mut middle).shared();
    let dependent = failed
        .clone()
        .then(|middle| {
            later.fetch_add(1, Ordering::SeqCst);
            async move {
                scale_add(pipeline, &**middle, output_view, 3.0, 0.0)
                    .await
                    .map(drop)
                    .map_err(Arc::new)
            }
        })
        .wait()
        .unwrap_err();
    let direct = failed.wait().expect_err("the rejected launch fails");
    assert!(Arc::ptr_eq(&dependent, &direct));
    assert!(matches!(*direct, Error::InvalidLaunch(_)), "{direct:?}");
    assert_eq!(later.load(Ordering::SeqCst), 0);
    assert!(output.read().wait().iter().all(|&value| value == -1.0));
    Ok(())
}
