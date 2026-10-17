use std::{
    future::{Future, IntoFuture, poll_fn},
    pin::{Pin, pin},
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use futures_util::future::Shared;

use super::{BatchOperation, Timed};
use crate::metal::sync::{Mutex, lock};

/// Lazy fallible work: every `IntoFuture` of a `Result`. Callbacks run on the polling thread;
/// with `futures::FutureExt` also in scope, call the methods as `Operation::map(work, f)`.
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

    /// Runs the operation once however many clones of the returned future await it, each getting
    /// the same `Arc` result; dropping all clones drops the operation, not its submitted work.
    fn shared(self) -> impl Future<Output = Result<Arc<Self::Value>, Arc<Self::Error>>> + Clone {
        let waker = Arc::new(Mutex::new(None));
        let seen = Arc::clone(&waker);
        let work = async move {
            let mut future = pin!(self.into_future());
            poll_fn(|cx| {
                record(&seen, cx.waker());
                future.as_mut().poll(cx)
            })
            .await
            .map(Arc::new)
            .map_err(Arc::new)
        };
        Consumer {
            shared: futures_util::FutureExt::shared(work),
            waker,
            polled: false,
        }
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

/// A clone of shared work. A clone dropped between polls wakes the waker the work last saw, so
/// clones that registered while a wake of the work was in flight take over polling.
struct Consumer<F: Future> {
    shared: Shared<F>,
    waker: Arc<Mutex<Option<Waker>>>,
    polled: bool,
}

impl<F: Future> Clone for Consumer<F> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            waker: Arc::clone(&self.waker),
            polled: false,
        }
    }
}

impl<F: Future<Output: Clone>> Future for Consumer<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        // Set before polling so that a panic in the work also wakes the other clones.
        self.polled = true;
        let poll = Pin::new(&mut self.shared).poll(cx);
        self.polled = poll.is_pending();
        poll
    }
}

impl<F: Future> Drop for Consumer<F> {
    fn drop(&mut self) {
        let waker = if self.polled {
            lock(&self.waker).clone()
        } else {
            None
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

fn record(seen: &Mutex<Option<Waker>>, waker: &Waker) {
    let mut seen = lock(seen);
    if !seen.as_ref().is_some_and(|known| known.will_wake(waker)) {
        *seen = Some(waker.clone());
    }
}

struct Unpark(std::thread::Thread);

impl Wake for Unpark {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
