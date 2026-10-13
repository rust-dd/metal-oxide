use std::{
    future::{Future, IntoFuture},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, ready},
    time::Duration,
};

use objc2::rc::autoreleasepool;

use super::{
    Argument, Pipeline,
    batch::Batch,
    completion::{Completion, GpuTime},
};
use crate::{DynamicLaunchConfig, Result};

mod sealed {
    pub trait Sealed {}
}

/// GPU work that runs when it is awaited or waited.
///
/// Nothing is encoded before the first poll or `wait`; then the whole operation encodes into
/// one command buffer and commits. A device has one ordered queue, so operations execute in
/// the order they are first polled or waited, not the order they were created. Dropping an
/// operation before that encodes nothing. Dropping its future later does not cancel the GPU
/// work: its resources stay retained until completion, and CPU access to its buffers waits.
///
/// Errors from pipeline creation, validation or encoding discard the whole command buffer,
/// and owned arguments are dropped with it. [`sequence`] runs several operations in one
/// command buffer.
pub trait Operation: sealed::Sealed + Sized {
    /// What the operation returns after its GPU work has finished.
    type Output;

    #[doc(hidden)]
    fn encode(self, batch: &mut Batch) -> Result<Self::Output>;

    /// Also returns the GPU execution time of the operation's command buffer.
    fn timed(self) -> Timed<Self> {
        Timed(self)
    }

    /// Runs the operation and blocks the calling thread until its GPU work has finished.
    fn wait(self) -> Result<Self::Output> {
        run(self).map(|(output, _)| output)
    }
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
    pipeline: Result<Pipeline>,
    config: DynamicLaunchConfig,
    arguments: A,
    bind: for<'a> fn(&'a mut A) -> Result<[Argument<'a>; N]>,
}

impl<A, const N: usize> Launch<A, N> {
    /// Describes a dispatch of `pipeline` whose kernel parameters `bind` builds from
    /// `arguments`. A pipeline creation error is returned when the operation runs.
    ///
    /// # Safety
    ///
    /// The bound arguments must match the kernel's types, slots, lengths, and access modes.
    /// The kernel must stay in bounds, respect read-only resources, have no data races, and
    /// leave valid values in every written element. Its grid and block must satisfy all
    /// participation and synchronization requirements. Dispatches of one command buffer run
    /// in order, and no external GPU or CPU access may conflict with them.
    pub unsafe fn new(
        pipeline: Result<&Pipeline>,
        config: impl Into<DynamicLaunchConfig>,
        arguments: A,
        bind: for<'a> fn(&'a mut A) -> Result<[Argument<'a>; N]>,
    ) -> Self {
        Self {
            pipeline: pipeline.cloned(),
            config: config.into(),
            arguments,
            bind,
        }
    }
}

impl<A, const N: usize> Operation for Launch<A, N> {
    type Output = A;

    fn encode(mut self, batch: &mut Batch) -> Result<A> {
        let pipeline = self.pipeline?;
        let arguments = (self.bind)(&mut self.arguments)?;
        // SAFETY: the creator of this launch upholds the contract of `Launch::new`.
        unsafe { batch.launch(&pipeline, self.config, &arguments)? };
        drop(arguments);
        Ok(self.arguments)
    }
}

/// Builds one operation from the operations `encode` pushes, in order, into a single command
/// buffer. Later operations see the writes of earlier ones; the closure runs when the
/// operation is first polled or waited, before the command buffer commits.
pub fn sequence<F, T>(encode: F) -> Sequence<F>
where
    F: FnOnce(&mut Batch) -> Result<T>,
{
    Sequence(encode)
}

/// Operations encoded into one command buffer; see [`sequence`].
#[must_use = "operations do nothing until awaited or waited"]
pub struct Sequence<F>(F);

impl<F, T> Operation for Sequence<F>
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

/// An operation that also returns its command buffer's GPU execution time; see
/// [`Operation::timed`]. The time is `None` when nothing was dispatched or the device
/// reports no timestamps.
#[must_use = "operations do nothing until awaited or waited"]
pub struct Timed<O>(O);

impl<O: Operation> Timed<O> {
    /// Runs the operation and blocks the calling thread until its GPU work has finished.
    pub fn wait(self) -> Result<(O::Output, Option<Duration>)> {
        run(self.0)
    }
}

/// The future of an operation; see [`Operation`].
#[must_use = "futures do nothing unless polled"]
pub struct Pending<O: Operation>(Run<O>);

/// The future of a [`Timed`] operation.
#[must_use = "futures do nothing unless polled"]
pub struct TimedPending<O: Operation>(Run<O>);

enum Run<O: Operation> {
    Unpolled(O),
    Running {
        completion: Arc<Completion>,
        output: O::Output,
    },
    Finished,
}

impl<O: Operation> Run<O> {
    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Result<(O::Output, GpuTime)>> {
        if let Self::Unpolled(_) = self {
            let Self::Unpolled(operation) = std::mem::replace(self, Self::Finished) else {
                unreachable!()
            };
            let (completion, output) = start(operation)?;
            *self = Self::Running { completion, output };
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

/// Encodes and commits an operation; the command buffer runs without the caller.
fn start<O: Operation>(operation: O) -> Result<(Arc<Completion>, O::Output)> {
    autoreleasepool(|_| {
        let mut batch = Batch::new();
        let output = operation.encode(&mut batch)?;
        Ok((batch.commit(), output))
    })
}

fn run<O: Operation>(operation: O) -> Result<(O::Output, GpuTime)> {
    let (completion, output) = start(operation)?;
    Ok((output, completion.wait()?))
}

// The futures never pin their fields: the operation moves out on the first poll.
impl<O: Operation> Unpin for Pending<O> {}
impl<O: Operation> Unpin for TimedPending<O> {}

impl<O: Operation> Future for Pending<O> {
    type Output = Result<O::Output>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut()
            .0
            .poll(cx)
            .map(|result| result.map(|(output, _)| output))
    }
}

impl<O: Operation> Future for TimedPending<O> {
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

impl<O: Operation> IntoFuture for Timed<O> {
    type Output = Result<(O::Output, Option<Duration>)>;
    type IntoFuture = TimedPending<O>;

    fn into_future(self) -> TimedPending<O> {
        TimedPending(Run::Unpolled(self.0))
    }
}
