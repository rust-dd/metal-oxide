use std::{
    future::{Future, IntoFuture},
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use super::{BatchOperation, Timed};

/// Lazy fallible work: a launch, a sequence, or a chain that continues one. Callbacks run on
/// the polling thread after the work before them succeeds.
pub trait Operation: IntoFuture<Output = Result<Self::Value, Self::Error>> + Sized {
    type Value;
    type Error;

    /// Processes the value once it is ready; an error skips `process`.
    fn map<F, T>(self, process: F) -> impl Future<Output = Result<T, Self::Error>>
    where
        F: FnOnce(Self::Value) -> T,
    {
        async move { self.into_future().await.map(process) }
    }

    /// Starts and awaits dependent work once this operation succeeds; an error skips `next`.
    fn then<F, N, T>(self, next: F) -> impl Future<Output = Result<T, Self::Error>>
    where
        F: FnOnce(Self::Value) -> N,
        N: IntoFuture<Output = Result<T, Self::Error>>,
    {
        async move { next(self.into_future().await?).into_future().await }
    }

    /// Runs the operation once for every clone of the returned future, each of which gets the
    /// same `Arc` value or error; dropping all clones drops unfinished work.
    fn shared(self) -> impl Future<Output = Result<Arc<Self::Value>, Arc<Self::Error>>> + Clone {
        futures_util::FutureExt::shared(async move {
            self.into_future().await.map(Arc::new).map_err(Arc::new)
        })
    }

    /// Also returns the GPU execution time of the operation's command buffer.
    fn timed(self) -> Timed<Self>
    where
        Self: BatchOperation,
    {
        Timed(self)
    }

    /// Runs the operation and blocks the calling thread until it completes.
    fn wait(self) -> Result<Self::Value, Self::Error> {
        let waker = Waker::from(Arc::new(Unpark(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut future = pin!(self.into_future());
        loop {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(result) => return result,
                Poll::Pending => std::thread::park(),
            }
        }
    }
}

impl<O, T, E> Operation for O
where
    O: IntoFuture<Output = Result<T, E>>,
{
    type Value = T;
    type Error = E;
}

struct Unpark(std::thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
