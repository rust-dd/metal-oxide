#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use support::{Project, checked};

const HOST: &str = r#"
use std::{
    future::{Future, IntoFuture},
    panic::{AssertUnwindSafe, catch_unwind},
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};
use metal_oxide::{Device, Dim3, Error, LaunchConfig, Operation, sequence};

#[metal_oxide::kernels]
mod kernels {
    use metal_oxide::device::{DisjointBuffer, thread_index};

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Value { pub word: u32 }

    #[kernel]
    pub fn set(mut output: DisjointBuffer<Value>) {
        if let Some(value) = output.get_mut(&thread_index()) {
            value.word = 42;
        }
    }
}

struct Unpark(std::thread::Thread);
impl Wake for Unpark {
    fn wake(self: Arc<Self>) { self.0.unpark(); }
}

fn block_on<F: IntoFuture>(operation: F) -> F::Output {
    let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
    let mut future = pin!(operation.into_future());
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(result) => return result,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn main() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let mut output = device.buffer_from_slice(&[kernels::Value { word: 7 }])?;
    let config = LaunchConfig::<1>::for_elements(1)?;
    let buffer = &mut output;
    let buffer = sequence(|batch| {
        let (buffer,) = batch.push(kernels::set(&device, config, buffer))?;
        assert!(catch_unwind(AssertUnwindSafe(|| buffer.read().wait())).is_err());
        Ok(&*buffer)
    }).wait()?;
    let first = buffer.read().wait();
    let second = buffer.read().wait();
    assert_eq!(first[0].word, 42);
    assert_eq!(second[0].word, 42);
    assert_eq!(first.as_ptr(), second.as_ptr());

    let other = Device::system_default()?;
    let mut output = other.buffer_from_slice(&[kernels::Value { word: 9 }])?;
    block_on(kernels::set(&other, config, &mut output))?;
    assert_eq!(output.read().wait()[0].word, 42);
    output.write().wait()[0].word = 11;
    block_on(kernels::set(&other, config, &mut output))?;
    assert_eq!(output.read().wait()[0].word, 42);

    let empty = LaunchConfig::<1, 2>::new(Dim3::x(0));
    assert!(matches!(
        block_on(kernels::set(&other, empty, &mut output)),
        Err(Error::InvalidLaunch(_))
    ));
    assert_eq!(output.read().wait()[0].word, 42);
    Ok(())
}
"#;

#[test]
#[ignore = "requires a Metal device and the kernel toolchain"]
fn safe_generated_launches_reserve_buffers_and_await_cold_pipelines() {
    let project = Project::new(HOST);
    checked(project.cargo(&["run", "--quiet"]).output().unwrap());
}
