use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll, ready},
};

use super::Buffer;
use crate::GpuValue;

/// The values of a buffer once its pending GPU work has finished; see [`Buffer::read`].
#[must_use = "futures do nothing unless polled or waited"]
pub struct Read<'a, T: GpuValue>(&'a Buffer<T>);

/// The values of a buffer for CPU writes once its pending GPU work has finished; see
/// [`Buffer::write`].
#[must_use = "futures do nothing unless polled or waited"]
pub struct Write<'a, T: GpuValue>(Option<&'a mut Buffer<T>>);

impl<'a, T: GpuValue> Read<'a, T> {
    pub(super) fn new(buffer: &'a Buffer<T>) -> Self {
        Self(buffer)
    }

    /// Blocks the calling thread until pending GPU work on the buffer has finished.
    ///
    /// # Panics
    ///
    /// Panics while an uncommitted batch has recorded a write to this buffer.
    pub fn wait(self) -> &'a [T] {
        self.0.access.synchronize(false);
        // SAFETY: no GPU write is recorded or pending, and new writes need `&mut`.
        unsafe { self.0.values() }
    }
}

impl<'a, T: GpuValue> Future for Read<'a, T> {
    type Output = &'a [T];

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<&'a [T]> {
        let buffer = self.0;
        ready!(buffer.access.poll(false, cx));
        // SAFETY: no GPU write is recorded or pending, and new writes need `&mut`.
        Poll::Ready(unsafe { buffer.values() })
    }
}

impl<'a, T: GpuValue> Write<'a, T> {
    pub(super) fn new(buffer: &'a mut Buffer<T>) -> Self {
        Self(Some(buffer))
    }

    /// Blocks the calling thread until pending GPU work on the buffer has finished.
    ///
    /// # Panics
    ///
    /// Panics while an uncommitted batch has recorded this buffer.
    pub fn wait(mut self) -> &'a mut [T] {
        let buffer = self.0.take().unwrap();
        buffer.access.synchronize(true);
        // SAFETY: no recording exists, every pending access finished, and `&mut` excludes new ones.
        unsafe { buffer.values_mut() }
    }
}

impl<'a, T: GpuValue> Future for Write<'a, T> {
    type Output = &'a mut [T];

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<&'a mut [T]> {
        let this = self.get_mut();
        let buffer = this
            .0
            .as_ref()
            .expect("a buffer write future was polled after it completed");
        ready!(buffer.access.poll(true, cx));
        let buffer = this.0.take().unwrap();
        // SAFETY: no recording exists, every pending access finished, and `&mut` excludes new ones.
        Poll::Ready(unsafe { buffer.values_mut() })
    }
}
