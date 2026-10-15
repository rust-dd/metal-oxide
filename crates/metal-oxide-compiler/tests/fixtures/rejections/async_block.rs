#![no_std]

use core::{future::Future, pin::pin, task::{Context, Poll, Waker}};
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn poll_async_block(output: WriteBuffer<u32>, value: u32) {
    let mut future = pin!(async move { value });
    let mut context = Context::from_waker(Waker::noop());
    if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
        unsafe { output.store_unchecked(0, result) };
    }
}
