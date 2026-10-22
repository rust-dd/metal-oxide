use std::{
    future::poll_fn,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread::JoinHandle,
};

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTL4CommandQueue, MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLSharedEvent};
use tokio::runtime::Builder;

use super::*;
use crate::{Buffer, Dim3, LaunchConfig, Module, metal::backend::Backend};

struct Gate {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    timed_out: Arc<AtomicBool>,
    cancel: mpsc::Sender<()>,
    watchdog: Option<JoinHandle<()>>,
}

impl Gate {
    fn new(device: &Device) -> Self {
        let event = device.raw().newSharedEvent().unwrap();
        let timed_out = Arc::new(AtomicBool::new(false));
        let (cancel, receiver) = mpsc::channel();
        let watchdog = std::thread::spawn({
            let event = event.clone();
            let timed_out = Arc::clone(&timed_out);
            move || {
                if receiver.recv_timeout(Duration::from_secs(10)).is_err() {
                    timed_out.store(true, Ordering::SeqCst);
                    event.setSignaledValue(u64::MAX);
                }
            }
        });
        Self {
            event,
            timed_out,
            cancel,
            watchdog: Some(watchdog),
        }
    }

    fn block(&self, device: &Device, value: u64) {
        let event = ProtocolObject::from_ref(&*self.event);
        match device.backend() {
            Backend::Classic { queues, .. } => {
                for queue in queues {
                    let command = autoreleasepool(|_| queue.native.raw.commandBuffer()).unwrap();
                    command.encodeWaitForEvent_value(event, value);
                    command.commit();
                }
            }
            Backend::Metal4 { queues, .. } => {
                for queue in queues {
                    queue.native.raw.waitForEvent_value(event, value);
                }
            }
        }
    }

    fn release_from_task(&self, pending: Arc<AtomicUsize>) -> tokio::task::JoinHandle<()> {
        let event = self.event.clone();
        tokio::spawn(async move {
            for value in 1..=2 {
                while pending.load(Ordering::SeqCst) < value {
                    tokio::task::yield_now().await;
                }
                event.setSignaledValue(value as u64);
            }
        })
    }

    fn check(&self, task: tokio::task::JoinHandle<()>) {
        task.abort();
        assert!(
            !self.timed_out.load(Ordering::SeqCst),
            "GPU waiting blocked the executor"
        );
        assert_eq!(self.event.signaledValue(), 2);
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        // A failed assertion must also release native commands and the watchdog.
        self.event.setSignaledValue(u64::MAX);
        let _ = self.cancel.send(());
        self.watchdog.take().unwrap().join().unwrap();
    }
}

fn setup() -> Result<(Device, Pipeline, Buffer<u32>)> {
    let device = Device::system_default()?;
    let module = Module::from_source(
        &device,
        r#"
        #include <metal_stdlib>
        using namespace metal;
        kernel void increment(device uint* value [[buffer(0)]]) { value[0] += 1; }
    "#,
    )?;
    let pipeline = Pipeline::new(&device, &module, "increment")?;
    let buffer = device.buffer_from_slice(&[0_u32])?;
    Ok((device, pipeline, buffer))
}

fn increment<'a>(
    pipeline: &Pipeline,
    buffer: &'a mut Buffer<u32>,
) -> Launch<&'a mut Buffer<u32>, 1> {
    // SAFETY: exactly one thread increments the buffer's initialized element.
    unsafe {
        Launch::new(
            Ok(pipeline),
            LaunchConfig::<1>::new(Dim3::x(1)),
            buffer,
            |buffer| Ok([Argument::read_write(buffer)]),
        )
    }
}

async fn observe<A, const N: usize>(
    launch: Launch<A, N>,
    pending: &AtomicUsize,
    stage: usize,
) -> Result<A> {
    let mut future = launch.into_future();
    poll_fn(|cx| {
        let result = Pin::new(&mut future).poll(cx);
        if result.is_pending() && matches!(future.0, Run::Running { .. }) {
            pending.store(stage, Ordering::SeqCst);
        }
        result
    })
    .await
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn awaiting_a_chain_never_blocks_a_current_thread_runtime() -> Result<()> {
    let (device, pipeline, mut buffer) = setup()?;
    let gate = Gate::new(&device);
    gate.block(&device, 1);
    Builder::new_current_thread().build()?.block_on(async {
        let pending = Arc::new(AtomicUsize::new(0));
        let task = gate.release_from_task(Arc::clone(&pending));
        let value = observe(increment(&pipeline, &mut buffer), &pending, 1)
            .then(|buffer| {
                gate.block(&device, 2);
                observe(increment(&pipeline, buffer), &pending, 2)
            })
            .then(|buffer| async move { Ok(buffer.read().await[0]) })
            .map(|value| value * 3)
            .await;
        gate.check(task);
        assert_eq!(value?, 6);
        Ok(())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn waiting_for_the_gpu_never_blocks_a_current_thread_runtime() -> Result<()> {
    let (device, pipeline, mut buffer) = setup()?;
    let gate = Gate::new(&device);
    gate.block(&device, 1);
    Builder::new_current_thread().build()?.block_on(async {
        let pending = Arc::new(AtomicUsize::new(0));
        let task = gate.release_from_task(Arc::clone(&pending));
        observe(increment(&pipeline, &mut buffer), &pending, 1).await?;
        gate.block(&device, 2);
        let mut launch = increment(&pipeline, &mut buffer).into_future();
        poll_fn(|cx| {
            assert!(Pin::new(&mut launch).poll(cx).is_pending());
            if matches!(launch.0, Run::Running { .. }) {
                Poll::Ready(())
            } else {
                Poll::Pending
            }
        })
        .await;
        drop(launch);
        let mut read = buffer.read();
        let values = poll_fn(|cx| {
            let result = Pin::new(&mut read).poll(cx);
            if result.is_pending() {
                pending.store(2, Ordering::SeqCst);
            }
            result
        })
        .await;
        gate.check(task);
        assert_eq!(values, [2]);
        Ok(())
    })
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn formatting_a_buffer_does_not_wait_for_gpu_completion() -> Result<()> {
    let (device, pipeline, mut buffer) = setup()?;
    let gate = Gate::new(&device);
    gate.block(&device, 1);
    let before = format!("{buffer:?}");
    let mut pending = increment(&pipeline, &mut buffer).into_future();
    assert!(
        Pin::new(&mut pending)
            .poll(&mut Context::from_waker(std::task::Waker::noop()))
            .is_pending()
    );
    assert!(matches!(pending.0, Run::Running { .. }));
    drop(pending);
    assert_eq!(format!("{buffer:?}"), before);
    assert!(
        !gate.timed_out.load(Ordering::SeqCst),
        "formatting waited for the GPU"
    );
    gate.event.setSignaledValue(1);
    assert_eq!(buffer.read().wait(), [1]);
    Ok(())
}
