use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    task::{Context, Waker},
};

use crate::stages;
use metal_oxide::{Device, LaunchConfig, Operation};

#[test]
#[ignore = "requires a Metal device"]
fn async_generated_kernel_chain_matches_cpu() {
    super::main().unwrap();
}

#[test]
#[ignore = "requires a Metal device"]
fn cancelled_generated_operation_keeps_buffers_synchronized() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let n = 1_000_003;
    let input = device.buffer_from_slice(&vec![2.0_f32; n])?;
    let mut output = device.buffer_zeroed::<f32>(n)?;
    let config = LaunchConfig::<128>::for_elements(n as u32)?;
    // Warm the pipeline so cancellation follows submission on the first poll.
    // SAFETY: the grid has one writer per valid index and disjoint n-element buffers.
    unsafe { stages::scale_128(&device, config, &input, &mut output, n as u32) }.wait()?;
    output.write().wait().fill(0.0);
    // SAFETY: the specialized grid has one writer per valid index and disjoint n-element buffers.
    let operation = unsafe { stages::scale_128(&device, config, &input, &mut output, n as u32) };
    let mut future = operation.into_future();
    let _ = Pin::new(&mut future).poll(&mut Context::from_waker(Waker::noop()));
    drop(future);
    assert!(output.read().wait().iter().all(|&value| value == 7.0));
    Ok(())
}
