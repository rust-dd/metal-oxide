use std::{
    fmt,
    sync::Arc,
    task::{Context, Poll, Waker},
};

use dispatch2::{DispatchQueue, DispatchQueueGlobalPriority, GlobalQueueIdentifier};
use objc2::rc::autoreleasepool;

use crate::{
    Result,
    metal::sync::{Mutex, lock},
};

use super::{Device, Pipeline};
use crate::metal::Library;

pub(in crate::metal) enum Source {
    Ready(Pipeline),
    Embedded {
        device: Device,
        library: &'static Library,
        entry: &'static str,
        preparation: Option<Preparation>,
    },
}

impl fmt::Debug for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready(pipeline) => pipeline.fmt(f),
            Self::Embedded {
                device,
                entry,
                preparation,
                ..
            } => f
                .debug_struct("Embedded")
                .field("entrypoint", entry)
                .field("device_id", &device.id())
                .field("preparing", &preparation.is_some())
                .finish_non_exhaustive(),
        }
    }
}

impl Source {
    pub(in crate::metal) fn ready(pipeline: Pipeline) -> Self {
        Self::Ready(pipeline)
    }

    pub(in crate::metal) fn embedded(
        device: &Device,
        library: &'static Library,
        entry: &'static str,
    ) -> Self {
        Self::Embedded {
            device: device.clone(),
            library,
            entry,
            preparation: None,
        }
    }

    pub(in crate::metal) fn device(&self) -> &Device {
        match self {
            Self::Ready(pipeline) => pipeline.device(),
            Self::Embedded { device, .. } => device,
        }
    }

    pub(in crate::metal) fn poll(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let Self::Embedded {
            device,
            library,
            entry,
            preparation,
        } = self
        else {
            return Poll::Ready(Ok(()));
        };
        if preparation.is_none() {
            if let Some(pipeline) = device.cached_kernel(library, entry) {
                *self = Self::Ready(pipeline);
                return Poll::Ready(Ok(()));
            }
            let device = device.clone();
            let (library, entry) = (*library, *entry);
            *preparation = Some(Preparation::new(move || device.kernel(library, entry)));
        }
        match preparation.as_ref().unwrap().poll(cx) {
            Poll::Ready(Ok(pipeline)) => {
                *self = Self::Ready(pipeline);
                Poll::Ready(Ok(()))
            }
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }

    pub(in crate::metal) fn pipeline(&self) -> &Pipeline {
        let Self::Ready(pipeline) = self else {
            panic!("pipeline preparation has not completed");
        };
        pipeline
    }
}

pub(in crate::metal) struct Preparation(Arc<Mutex<State>>);

struct State {
    result: Option<Result<Pipeline>>,
    waker: Option<Waker>,
}

impl Preparation {
    fn new(work: impl FnOnce() -> Result<Pipeline> + Send + 'static) -> Self {
        let preparation = Self(Arc::new(Mutex::new(State {
            result: None,
            waker: None,
        })));
        let worker = Self(Arc::clone(&preparation.0));
        DispatchQueue::global_queue(GlobalQueueIdentifier::Priority(
            DispatchQueueGlobalPriority::Default,
        ))
        .exec_async(move || {
            let result = autoreleasepool(|_| work());
            worker.complete(result);
        });
        preparation
    }

    fn complete(&self, result: Result<Pipeline>) {
        let waker = {
            let mut state = lock(&self.0);
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(waker) = waker {
            // User wakers must not unwind through GCD's native callback.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }

    fn poll(&self, cx: &mut Context<'_>) -> Poll<Result<Pipeline>> {
        let mut state = lock(&self.0);
        if let Some(result) = state.result.take() {
            return Poll::Ready(result);
        }
        state.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[cfg(test)]
mod tests;
