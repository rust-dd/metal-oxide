use std::{
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use metal_oxide::{Device, LaunchConfig, sequence};

struct Unpark(std::thread::Thread);
impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}

pub(super) fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub(super) async fn verify() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::system_default()?;
    let kernels = super::kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    for block in [128, 256] {
        for n in [0_u32, 1, 255, 256, 257, 1_000_003] {
            let values = (0..n).map(|i| (i % 1024) as f32 * 0.25).collect::<Vec<_>>();
            let input = device.buffer_from_slice(&values)?;
            let mut intermediate = device.buffer_zeroed::<f32>(n as usize)?;
            let mut output = device.buffer_zeroed::<f32>(n as usize)?;
            // SAFETY: both stages guard n, use disjoint allocations, and run in encoded order.
            sequence(|batch| unsafe {
                if block == 128 {
                    batch.push(kernels.scale_128(
                        LaunchConfig::<128>::for_elements(n)?,
                        &input,
                        &mut intermediate,
                        n,
                    ))?;
                } else {
                    batch.push(kernels.scale_256(
                        LaunchConfig::<256>::for_elements(n)?,
                        &input,
                        &mut intermediate,
                        n,
                    ))?;
                }
                batch.push(kernels.finish(
                    LaunchConfig::<256>::for_elements(n)?,
                    &intermediate,
                    &mut output,
                    n,
                ))?;
                Ok(())
            })
            .await?;
            let scale = if block == 128 { 2.0 } else { 4.0 };
            for (i, &actual) in output.read().wait().iter().enumerate() {
                assert_eq!(
                    actual,
                    (values[i] * scale + 3.0) * 5.0 + 7.0,
                    "block={block}, n={n}, i={i}"
                );
            }
            println!(
                "pipeline block={block}, n={n}: verified on {}",
                device.name()
            );
        }
    }
    Ok(())
}
