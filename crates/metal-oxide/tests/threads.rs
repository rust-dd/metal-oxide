#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::{
    future::Future,
    pin::pin,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
};

use metal_oxide::{
    Argument, Buffer, Device, Error, GpuValue, LaunchConfig, Layout, Module, Pipeline, Result,
    Submission, SubmissionReport,
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
kernel void write_values(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = 10 + i;
}
"#;

fn setup() -> Result<(Device, Module, Pipeline)> {
    let device = Device::system_default()?;
    let module = Module::from_source(&device, SOURCE)?;
    let pipeline = Pipeline::new(&device, &module, "scale_add")?;
    Ok((device, module, pipeline))
}

/// Submits `output = input * scale + offset` over the first n elements.
///
/// # Safety
///
/// Both buffers must hold at least n elements.
unsafe fn scale_add<'a>(
    device: &'a Device,
    pipeline: &'a Pipeline,
    input: &'a Buffer<f32>,
    output: &'a mut Buffer<f32>,
    [scale, offset]: [f32; 2],
    n: u32,
) -> Result<Submission<'a>> {
    // SAFETY: the caller sizes both buffers; the kernel guards the padded grid.
    unsafe {
        device.submit(move |batch| {
            batch.launch(
                pipeline,
                LaunchConfig::<256>::for_elements(n)?,
                &[
                    Argument::read(input),
                    Argument::write(output),
                    Argument::value(scale)?,
                    Argument::value(offset)?,
                    Argument::value(n)?,
                ],
            )
        })
    }
}

struct Unpark(thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Unpark(thread::current())));
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn runtime_handles_are_send_and_sync() {
    assert_send_sync::<Device>();
    assert_send_sync::<Module>();
    assert_send_sync::<Pipeline>();
    assert_send_sync::<Buffer<f32>>();
    assert_send_sync::<Buffer<(u32, f32)>>();
    assert_send_sync::<Submission<'static>>();
    assert_send_sync::<Argument<'static>>();
    assert_send_sync::<Error>();
    assert_send_sync::<SubmissionReport>();
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn threads_share_one_device_module_and_pipeline() -> Result<()> {
    let (device, _module, pipeline) = setup()?;
    thread::scope(|scope| {
        let workers = (0..8)
            .map(|worker| {
                let (device, pipeline) = (&device, &pipeline);
                scope.spawn(move || -> Result<()> {
                    for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
                        let values = (0..n).map(|i| i as f32).collect::<Vec<_>>();
                        let input = device.buffer_from_slice(&values)?;
                        let mut output = device.buffer_zeroed::<f32>(n as usize)?;
                        let offset = worker as f32;
                        // SAFETY: both buffers hold n elements.
                        unsafe {
                            scale_add(device, pipeline, &input, &mut output, [2.0, offset], n)
                        }?
                        .wait()?;
                        for (i, value) in output.as_slice().iter().enumerate() {
                            assert_eq!(*value, i as f32 * 2.0 + offset, "worker {worker}, n {n}");
                        }
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .try_for_each(|worker| worker.join().unwrap())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn concurrent_launches_and_cpu_views_share_a_read_only_input() -> Result<()> {
    let (device, _module, pipeline) = setup()?;
    let n = 65_537_u32;
    let input = device.buffer_from_slice(&vec![3.0_f32; n as usize])?;
    thread::scope(|scope| {
        let launches = (0..4)
            .map(|worker| {
                let (device, pipeline, input) = (&device, &pipeline, &input);
                scope.spawn(move || -> Result<()> {
                    let mut output = device.buffer_zeroed::<f32>(n as usize)?;
                    for round in 0..8 {
                        let offset = (worker * 8 + round) as f32;
                        // SAFETY: both buffers hold n elements.
                        unsafe {
                            scale_add(device, pipeline, input, &mut output, [1.0, offset], n)
                        }?
                        .wait()?;
                        assert!(output.as_slice().iter().all(|&value| value == 3.0 + offset));
                    }
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        let readers = (0..4)
            .map(|_| {
                let input = &input;
                scope.spawn(move || {
                    for _ in 0..64 {
                        assert!(input.as_slice().iter().all(|&value| value == 3.0));
                    }
                })
            })
            .collect::<Vec<_>>();
        readers
            .into_iter()
            .for_each(|reader| reader.join().unwrap());
        launches
            .into_iter()
            .try_for_each(|launch| launch.join().unwrap())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn buffers_written_on_one_thread_synchronize_on_another() -> Result<()> {
    let (device, _module, pipeline) = setup()?;
    let pipeline = Arc::new(pipeline);
    let n = 1_000_003_u32;
    for forget in [false, true] {
        let (device, pipeline) = (device.clone(), Arc::clone(&pipeline));
        let output = thread::spawn(move || -> Result<Buffer<f32>> {
            let input = device.buffer_from_slice(&vec![2.0_f32; n as usize])?;
            let mut output = device.buffer_zeroed::<f32>(n as usize)?;
            // SAFETY: both buffers hold n elements.
            let submission =
                unsafe { scale_add(&device, &pipeline, &input, &mut output, [3.0, 1.0], n) }?;
            if forget {
                std::mem::forget(submission);
            } else {
                drop(submission);
            }
            Ok(output)
        })
        .join()
        .unwrap()?;
        assert!(output.as_slice().iter().all(|&value| value == 7.0));
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn submissions_complete_on_the_thread_that_awaits_them() -> Result<()> {
    let (device, _module, pipeline) = setup()?;
    let n = 1_000_003_u32;
    let input = device.buffer_from_slice(&vec![4.0_f32; n as usize])?;
    let mut waited = device.buffer_zeroed::<f32>(n as usize)?;
    let mut awaited = device.buffer_zeroed::<f32>(n as usize)?;
    // SAFETY: all buffers hold n elements.
    let first = unsafe { scale_add(&device, &pipeline, &input, &mut waited, [2.0, 1.0], n) }?;
    // SAFETY: all buffers hold n elements.
    let second = unsafe { scale_add(&device, &pipeline, &input, &mut awaited, [3.0, 2.0], n) }?;
    thread::scope(|scope| {
        let waiting = scope.spawn(move || first.wait());
        let awaiting = scope.spawn(move || block_on(second));
        waiting.join().unwrap()?;
        awaiting.join().unwrap()
    })?;
    assert!(waited.as_slice().iter().all(|&value| value == 9.0));
    assert!(awaited.as_slice().iter().all(|&value| value == 14.0));
    Ok(())
}

static DECODES: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq)]
struct Counted(u32);

impl GpuValue for Counted {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;
    fn layout() -> Result<Layout> {
        u32::layout()
    }
    fn zeroed() -> Self {
        Self(0)
    }
    fn encode(self, bytes: &mut [u8]) {
        self.0.encode(bytes);
    }
    fn decode(bytes: &[u8]) -> Self {
        DECODES.fetch_add(1, Ordering::Relaxed);
        Self(u32::decode(bytes))
    }
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn concurrent_codec_readbacks_decode_a_gpu_write_once() -> Result<()> {
    let (device, module, _) = setup()?;
    let pipeline = Pipeline::new(&device, &module, "write_values")?;
    let n = 4097_u32;
    let mut output = device.buffer_zeroed::<Counted>(n as usize)?;
    // SAFETY: the kernel writes each of the n initialized elements once.
    unsafe {
        device.launch(
            &pipeline,
            LaunchConfig::<64>::for_elements(n)?,
            &[Argument::write(&mut output), Argument::value(n)?],
        )?;
    }
    DECODES.store(0, Ordering::Relaxed);
    let barrier = Barrier::new(8);
    let expected = (0..n).map(|i| Counted(10 + i)).collect::<Vec<_>>();
    thread::scope(|scope| {
        for _ in 0..8 {
            let (output, barrier, expected) = (&output, &barrier, &expected);
            scope.spawn(move || {
                barrier.wait();
                assert_eq!(output.as_slice(), expected.as_slice());
            });
        }
    });
    assert_eq!(DECODES.load(Ordering::Relaxed), n as usize);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn runtime_objects_drop_on_another_thread_while_work_is_pending() -> Result<()> {
    let (device, module, pipeline) = setup()?;
    let n = 1_000_003_u32;
    let input = device.buffer_from_slice(&vec![1.0_f32; n as usize])?;
    let mut output = device.buffer_zeroed::<f32>(n as usize)?;
    // SAFETY: both buffers hold n elements; the passes run in order.
    let submission = unsafe {
        device.submit(|batch| {
            for pass in 0..64 {
                batch.launch(
                    &pipeline,
                    LaunchConfig::<256>::for_elements(n)?,
                    &[
                        Argument::read(&input),
                        Argument::write(&mut output),
                        Argument::value(1.0_f32)?,
                        Argument::value(pass as f32)?,
                        Argument::value(n)?,
                    ],
                )?;
            }
            Ok(())
        })?
    };
    std::mem::forget(submission);
    thread::spawn(move || drop((device, module, pipeline, input, output)))
        .join()
        .unwrap();
    let (device, _module, pipeline) = setup()?;
    let input = device.buffer_from_slice(&[5.0_f32; 3])?;
    let mut output = device.buffer_zeroed::<f32>(3)?;
    // SAFETY: both buffers hold three elements.
    unsafe { scale_add(&device, &pipeline, &input, &mut output, [2.0, 0.0], 3) }?.wait()?;
    assert_eq!(output.as_slice(), [10.0; 3]);
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn pipelines_are_created_concurrently_on_one_device() -> Result<()> {
    let device = Device::system_default()?;
    let source = (0..8)
        .map(|index| {
            format!(
                "kernel void fill_{index}(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ out[i] = i + {index}; }}\n"
            )
        })
        .collect::<String>();
    let module = Module::from_source(
        &device,
        &format!("#include <metal_stdlib>\nusing namespace metal;\n{source}"),
    )?;
    let barrier = Barrier::new(8);
    thread::scope(|scope| {
        let workers = (0..8_u32)
            .map(|index| {
                let (device, module, barrier) = (&device, &module, &barrier);
                scope.spawn(move || -> Result<()> {
                    barrier.wait();
                    let pipeline = Pipeline::new(device, module, &format!("fill_{index}"))?;
                    let mut output = device.buffer_zeroed::<u32>(32)?;
                    // SAFETY: each lane writes its own initialized output element.
                    unsafe {
                        device.launch(
                            &pipeline,
                            LaunchConfig::<32>::for_elements(32)?,
                            &[Argument::write(&mut output)],
                        )?;
                    }
                    assert_eq!(
                        output.as_slice(),
                        (0..32).map(|i| i + index).collect::<Vec<_>>()
                    );
                    Ok(())
                })
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .try_for_each(|worker| worker.join().unwrap())
    })
}
