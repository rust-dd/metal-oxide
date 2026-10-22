#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Wake, Waker},
    time::{Duration, Instant},
};

use metal_oxide::{
    Argument, Buffer, Device, Dim3, Error, Launch, LaunchConfig, Module, Operation, Pipeline,
    ReadArgument, Result, WriteArgument, sequence,
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

struct Kernels {
    scale_add: Pipeline,
    spin: Pipeline,
}

fn setup() -> Result<(Device, Kernels)> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let kernels = Kernels {
        scale_add: Pipeline::new(&device, &module, "scale_add")?,
        spin: Pipeline::new(&device, &module, "spin")?,
    };
    Ok((device, kernels))
}

/// `output = input * scale + offset` over the elements of `input`.
fn scale_add<I: ReadArgument<f32>, O: WriteArgument<f32>>(
    pipeline: &Pipeline,
    input: I,
    output: O,
    scale: f32,
    offset: f32,
) -> Launch<(I, O, f32, f32, u32), 5> {
    let n = input.buffer().len() as u32;
    let config = LaunchConfig::<256>::for_elements(n).unwrap();
    // SAFETY: callers pass equally long buffers in distinct allocations; the kernel guards n.
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

/// Each element iterates an LCG `rounds` times, which keeps the GPU busy.
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

/// Doubles the spin rounds until one spin takes at least `duration`.
fn calibrate(kernels: &Kernels, out: &mut Buffer<u32>, duration: Duration) -> Result<u32> {
    let mut rounds = 1 << 12;
    loop {
        let start = Instant::now();
        spin(&kernels.spin, out, rounds).wait()?;
        if start.elapsed() >= duration || rounds >= 1 << 28 {
            return Ok(rounds);
        }
        rounds *= 2;
    }
}

struct Unpark(std::thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

/// Polls `future` on this thread until `stop` holds after a pending poll.
fn poll_until<F: Future>(mut future: Pin<&mut F>, stop: impl Fn() -> bool) {
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    loop {
        let poll = future.as_mut().poll(&mut Context::from_waker(&waker));
        assert!(
            poll.is_pending(),
            "the chain completed before it was dropped"
        );
        if stop() {
            return;
        }
        std::thread::park();
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn borrowed_launches_feed_readback_and_host_processing() -> Result<()> {
    let (device, kernels) = setup()?;
    for n in [0, 257, 1_000_003] {
        let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
        let mut middle = device.buffer_zeroed::<f32>(n)?;
        let mut output = device.buffer_from_slice(&vec![-1.0_f32; n])?;
        let fifteens = scale_add(&kernels.scale_add, &input, &mut middle, 2.0, 1.0)
            .then(|(_, middle, ..)| scale_add(&kernels.scale_add, &*middle, &mut output, 3.0, 0.0))
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
    let (device, kernels) = setup()?;
    let runtime = Builder::new_multi_thread().worker_threads(4).build()?;
    let n = 65_537;
    runtime.block_on(async {
        let mut tasks = Vec::new();
        for task in 0..16_u8 {
            let input = device.buffer_from_slice(&vec![f32::from(task); n])?;
            let middle = device.buffer_zeroed::<f32>(n)?;
            let output = device.buffer_zeroed::<f32>(n)?;
            let pipeline = kernels.scale_add.clone();
            let chain = scale_add(&kernels.scale_add, input, middle, 2.0, 1.0)
                .then(move |(_, middle, ..)| scale_add(&pipeline, middle, output, 3.0, 0.0))
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
    let (device, kernels) = setup()?;
    let n = 1_000_003;
    let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
    let mut middle = device.buffer_zeroed::<f32>(n)?;
    let mut output = device.buffer_from_slice(&vec![-1.0_f32; n])?;
    let later = AtomicUsize::new(0);
    let result = scale_add(&kernels.scale_add, &input, &mut middle, 2.0, 1.0)
        .then(|(_, middle, ..)| rejected(&kernels.scale_add, middle))
        .then(|middle| {
            later.fetch_add(1, Ordering::SeqCst);
            scale_add(&kernels.scale_add, &*middle, &mut output, 3.0, 0.0)
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
fn dropped_chains_start_no_later_stage_and_keep_cpu_access_synchronized() -> Result<()> {
    let (device, kernels) = setup()?;
    let n = 32_768;
    let mut busy = device.buffer_zeroed::<u32>(n)?;
    let rounds = calibrate(&kernels, &mut busy, Duration::from_millis(20))?;
    let spun = busy.read().wait().to_vec();
    let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
    let mut output = device.buffer_from_slice(&vec![-1.0_f32; n])?;
    let later = AtomicUsize::new(0);

    busy.write().wait().fill(0);
    let chain = spin(&kernels.spin, &mut busy, rounds).then(|_| {
        later.fetch_add(1, Ordering::SeqCst);
        scale_add(&kernels.scale_add, &input, &mut output, 1.0, 0.0)
    });
    drop(chain);
    assert!(busy.read().wait().iter().all(|&value| value == 0));

    let mut chain = Box::pin(spin(&kernels.spin, &mut busy, rounds).then(|_| {
        later.fetch_add(1, Ordering::SeqCst);
        scale_add(&kernels.scale_add, &input, &mut output, 1.0, 0.0)
    }));
    poll_until(chain.as_mut(), || true);
    drop(chain);
    assert_eq!(busy.read().wait(), spun);
    assert!(output.read().wait().iter().all(|&value| value == -1.0));

    busy.write().wait().fill(0);
    let started = AtomicUsize::new(0);
    let mut chain = Box::pin(
        scale_add(&kernels.scale_add, &input, &mut output, 2.0, 1.0)
            .then(|_| {
                started.fetch_add(1, Ordering::SeqCst);
                spin(&kernels.spin, &mut busy, rounds)
            })
            .map(|_| later.fetch_add(1, Ordering::SeqCst)),
    );
    poll_until(chain.as_mut(), || started.load(Ordering::SeqCst) == 1);
    drop(chain);
    assert_eq!(busy.read().wait(), spun);
    assert!(output.read().wait().iter().all(|&value| value == 5.0));
    assert_eq!(later.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn a_shared_launch_runs_once_for_concurrent_readback_and_dependent_launches() -> Result<()> {
    let (device, kernels) = setup()?;
    let n = 1_000_003;
    let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
    let mut middle = device.buffer_zeroed::<f32>(n)?;
    let mut tripled = device.buffer_from_slice(&vec![-1.0_f32; n])?;
    let (pipeline, tripled_view) = (&kernels.scale_add, &mut tripled);
    let encodes = AtomicUsize::new(0);
    let produced = sequence(|batch| {
        encodes.fetch_add(1, Ordering::SeqCst);
        batch.push(scale_add(&kernels.scale_add, &input, &mut middle, 2.0, 1.0))
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
    let (device, kernels) = setup()?;
    let mut middle = device.buffer_zeroed::<f32>(257)?;
    let mut output = device.buffer_from_slice(&vec![-1.0_f32; 257])?;
    let later = AtomicUsize::new(0);
    let (pipeline, output_view) = (&kernels.scale_add, &mut output);
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

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn dropping_every_clone_of_a_running_shared_launch_keeps_cpu_access_synchronized() -> Result<()> {
    let (device, kernels) = setup()?;
    let mut busy = device.buffer_zeroed::<u32>(32_768)?;
    let rounds = calibrate(&kernels, &mut busy, Duration::from_millis(20))?;
    let spun = busy.read().wait().to_vec();
    busy.write().wait().fill(0);
    let later = AtomicUsize::new(0);
    let shared = spin(&kernels.spin, &mut busy, rounds).shared();
    let mut consumer = Box::pin(shared.clone().map(|_| later.fetch_add(1, Ordering::SeqCst)));
    poll_until(consumer.as_mut(), || true);
    drop(shared);
    drop(consumer);
    assert_eq!(busy.read().wait(), spun);
    assert_eq!(later.load(Ordering::SeqCst), 0);
    Ok(())
}
