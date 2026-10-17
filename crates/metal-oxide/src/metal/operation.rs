use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
    time::Duration,
};

use objc2::rc::autoreleasepool;

use super::{
    Argument, Device, Library, Pipeline,
    batch::Batch,
    completion::{Completion, GpuTime},
    pipeline::Source,
};
use crate::{DynamicLaunchConfig, Result};

mod compose;
pub use compose::Operation;

mod sealed {
    pub trait Sealed {}
}

/// GPU work that can be recorded into one command buffer with [`Batch::push`].
pub trait BatchOperation: sealed::Sealed + Sized {
    /// What [`Batch::push`] and awaiting the operation return.
    type Output;

    #[doc(hidden)]
    fn encode(self, batch: &mut Batch) -> Result<Self::Output>;
}

/// One kernel dispatch. Its arguments, in the form they were passed, are its output.
///
/// A pending operation keeps its borrows, so no CPU view can observe a buffer it writes:
///
/// ```compile_fail
/// use std::future::IntoFuture;
/// use metal_oxide::{Argument, Buffer, Launch, LaunchConfig, Pipeline};
/// fn check(pipeline: &Pipeline, output: &mut Buffer<u32>) {
///     let config = LaunchConfig::<32>::for_elements(32).unwrap();
///     // SAFETY: never runs; the borrow conflict stops compilation.
///     let pending = unsafe {
///         Launch::new(Ok(pipeline), config, (&mut *output,), |(output,)| {
///             Ok([Argument::write(output)])
///         })
///     }
///     .into_future();
///     let values = output.read().wait();
///     drop(pending);
///     println!("{}", values.len());
/// }
/// ```
#[must_use = "operations do nothing until awaited or waited"]
pub struct Launch<A, const N: usize> {
    pipeline: Result<Source>,
    config: DynamicLaunchConfig,
    arguments: A,
    bind: for<'a> fn(&'a mut A) -> Result<[Argument<'a>; N]>,
}

impl<A, const N: usize> Launch<A, N> {
    /// A dispatch of `pipeline` with the parameters `bind` builds from `arguments`; a pipeline
    /// error is returned when the operation runs.
    ///
    /// # Safety
    ///
    /// The arguments must match the kernel's parameters, and the kernel must stay in bounds,
    /// race-free and synchronized on this grid, with no conflicting access to its buffers.
    pub unsafe fn new(
        pipeline: Result<&Pipeline>,
        config: impl Into<DynamicLaunchConfig>,
        arguments: A,
        bind: for<'a> fn(&'a mut A) -> Result<[Argument<'a>; N]>,
    ) -> Self {
        Self {
            pipeline: pipeline.cloned().map(Source::ready),
            config: config.into(),
            arguments,
            bind,
        }
    }

    /// A dispatch of `entry` from a library that the kernel macros embedded; the launch
    /// functions they generate call it.
    ///
    /// # Safety
    ///
    /// As for [`Launch::new`].
    #[doc(hidden)]
    pub unsafe fn kernel(
        device: &Device,
        library: &'static Library,
        entry: &'static str,
        config: impl Into<DynamicLaunchConfig>,
        arguments: A,
        bind: for<'a> fn(&'a mut A) -> Result<[Argument<'a>; N]>,
    ) -> Self {
        Self {
            pipeline: Ok(Source::embedded(device, library, entry)),
            config: config.into(),
            arguments,
            bind,
        }
    }
}

impl<A, const N: usize> BatchOperation for Launch<A, N> {
    type Output = A;

    fn encode(mut self, batch: &mut Batch) -> Result<A> {
        let pipeline = self.pipeline?;
        let arguments = (self.bind)(&mut self.arguments)?;
        // SAFETY: the creator of this launch upholds the contract of `Launch::new`.
        unsafe { batch.launch(pipeline, self.config, &arguments)? };
        drop(arguments);
        Ok(self.arguments)
    }
}

/// One operation from those `encode` pushes into a single command buffer, where a launch
/// waits for the earlier ones whose buffers it uses and the others run concurrently; the
/// closure runs when the operation is first polled or waited.
pub fn sequence<F, T>(encode: F) -> Sequence<F>
where
    F: FnOnce(&mut Batch) -> Result<T>,
{
    Sequence(encode)
}

/// Operations encoded into one command buffer; see [`sequence`].
#[must_use = "operations do nothing until awaited or waited"]
pub struct Sequence<F>(F);

impl<F, T> BatchOperation for Sequence<F>
where
    F: FnOnce(&mut Batch) -> Result<T>,
{
    type Output = T;

    fn encode(self, batch: &mut Batch) -> Result<T> {
        (self.0)(batch)
    }
}

impl<A, const N: usize> sealed::Sealed for Launch<A, N> {}
impl<F> sealed::Sealed for Sequence<F> {}

/// An operation that also returns its GPU time, `None` without dispatches or timestamps.
#[must_use = "operations do nothing until awaited or waited"]
pub struct Timed<O>(O);

/// The future of a [`Launch`] or [`Sequence`].
#[must_use = "futures do nothing unless polled"]
pub struct Pending<O: BatchOperation>(Run<O>);

/// The future of a [`Timed`] operation.
#[must_use = "futures do nothing unless polled"]
pub struct TimedPending<O: BatchOperation>(Run<O>);

enum Run<O: BatchOperation> {
    Unpolled(O),
    Preparing {
        batch: Batch,
        output: O::Output,
    },
    Running {
        completion: Arc<Completion>,
        output: O::Output,
    },
    Finished,
}

impl<O: BatchOperation> Run<O> {
    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Result<(O::Output, GpuTime)>> {
        if let Self::Unpolled(_) = self {
            let Self::Unpolled(operation) = std::mem::replace(self, Self::Finished) else {
                unreachable!()
            };
            let (batch, output) = autoreleasepool(|_| {
                let mut batch = Batch::new();
                let output = operation.encode(&mut batch)?;
                Ok::<_, crate::Error>((batch, output))
            })?;
            *self = Self::Preparing { batch, output };
        }
        if let Self::Preparing { batch, .. } = self {
            let result = ready!(autoreleasepool(|_| batch.poll_commit(cx)));
            let Self::Preparing { output, .. } = std::mem::replace(self, Self::Finished) else {
                unreachable!()
            };
            *self = Self::Running {
                completion: result?,
                output,
            };
        }
        let Self::Running { completion, .. } = self else {
            panic!("an operation future was polled after it completed");
        };
        let result = ready!(completion.poll(cx));
        let Self::Running { output, .. } = std::mem::replace(self, Self::Finished) else {
            unreachable!()
        };
        Poll::Ready(result.map(|time| (output, time)))
    }
}

// The futures never pin their fields: the operation moves out on the first poll.
impl<O: BatchOperation> Unpin for Pending<O> {}
impl<O: BatchOperation> Unpin for TimedPending<O> {}

impl<O: BatchOperation> Future for Pending<O> {
    type Output = Result<O::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut()
            .0
            .poll(cx)
            .map(|result| result.map(|(output, _)| output))
    }
}

impl<O: BatchOperation> Future for TimedPending<O> {
    type Output = Result<(O::Output, Option<Duration>)>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().0.poll(cx)
    }
}

impl<A, const N: usize> IntoFuture for Launch<A, N> {
    type Output = Result<A>;
    type IntoFuture = Pending<Self>;

    fn into_future(self) -> Pending<Self> {
        Pending(Run::Unpolled(self))
    }
}

impl<F, T> IntoFuture for Sequence<F>
where
    F: FnOnce(&mut Batch) -> Result<T>,
{
    type Output = Result<T>;
    type IntoFuture = Pending<Self>;

    fn into_future(self) -> Pending<Self> {
        Pending(Run::Unpolled(self))
    }
}

impl<O: BatchOperation> IntoFuture for Timed<O> {
    type Output = Result<(O::Output, Option<Duration>)>;
    type IntoFuture = TimedPending<O>;

    fn into_future(self) -> TimedPending<O> {
        TimedPending(Run::Unpolled(self.0))
    }
}
