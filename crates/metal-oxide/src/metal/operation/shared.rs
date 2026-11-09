use std::{
    future::Future,
    mem,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use crate::metal::sync::{Mutex, lock};

/// Work that runs once for every clone: the clone that polls it while it is idle drives it, the
/// others wait for its wakes.
pub(super) struct Shared<F: Future> {
    work: Arc<Work<F>>,
    /// This clone's key among the waiting clones, from its first pending poll.
    id: Option<usize>,
}

struct Work<F: Future> {
    stage: Mutex<Stage<F>>,
    waiters: Arc<Waiters>,
}

enum Stage<F: Future> {
    Idle(Pin<Box<F>>),
    /// A clone is polling the future.
    Polling,
    Done(F::Output),
    Panicked,
}

/// The clones waiting for the work, woken by every wake of the work.
#[derive(Default)]
struct Waiters(Mutex<Registry>);

#[derive(Default)]
struct Registry {
    /// The waker of each clone's latest pending poll, until the next wake of the work.
    wakers: Vec<(usize, Waker)>,
    next: usize,
    /// The work was woken after its latest poll started.
    woken: bool,
}

impl<F: Future> Shared<F> {
    pub(super) fn new(future: F) -> Self {
        let work = Work {
            stage: Mutex::new(Stage::Idle(Box::pin(future))),
            waiters: Arc::default(),
        };
        Self {
            work: Arc::new(work),
            id: None,
        }
    }

    fn drive(&mut self, mut future: Pin<Box<F>>) -> Poll<F::Output>
    where
        F::Output: Clone,
    {
        let waker = Waker::from(Arc::clone(&self.work.waiters));
        let unwind = Unwind(&self.work);
        let poll = future.as_mut().poll(&mut Context::from_waker(&waker));
        mem::forget(unwind);
        match poll {
            Poll::Pending => {
                *lock(&self.work.stage) = Stage::Idle(future);
                Poll::Pending
            }
            Poll::Ready(output) => {
                *lock(&self.work.stage) = Stage::Done(output.clone());
                self.work.waiters.notify(self.id);
                Poll::Ready(output)
            }
        }
    }
}

impl<F: Future> Clone for Shared<F> {
    fn clone(&self) -> Self {
        Self {
            work: Arc::clone(&self.work),
            id: None,
        }
    }
}

impl<F: Future<Output: Clone>> Future for Shared<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        let mut stage = lock(&this.work.stage);
        let future = match mem::replace(&mut *stage, Stage::Polling) {
            Stage::Idle(future) => future,
            Stage::Polling => {
                // Registered under the stage lock, so the driver's completion wakes this clone.
                this.work.waiters.wait(&mut this.id, cx.waker());
                return Poll::Pending;
            }
            Stage::Done(output) => {
                let value = output.clone();
                *stage = Stage::Done(output);
                return Poll::Ready(value);
            }
            Stage::Panicked => {
                *stage = Stage::Panicked;
                drop(stage);
                panic!("the shared work panicked");
            }
        };
        // The driver waits too: a wake during its poll makes its task poll again.
        this.work.waiters.drive(&mut this.id, cx.waker());
        drop(stage);
        this.drive(future)
    }
}

impl<F: Future> Drop for Shared<F> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.work.waiters.leave(id);
        }
    }
}

/// Marks the work panicked and wakes the waiting clones when polling it unwinds.
struct Unwind<'a, F: Future>(&'a Work<F>);

impl<F: Future> Drop for Unwind<'_, F> {
    fn drop(&mut self) {
        *lock(&self.0.stage) = Stage::Panicked;
        self.0.waiters.notify(None);
    }
}

impl Waiters {
    /// Records the waker of a clone's pending poll.
    fn wait(&self, id: &mut Option<usize>, waker: &Waker) {
        lock(&self.0).insert(id, waker);
    }

    /// Records the driver's waker; its poll of the work consumes the earlier wakes.
    fn drive(&self, id: &mut Option<usize>, waker: &Waker) {
        let mut registry = lock(&self.0);
        registry.woken = false;
        registry.insert(id, waker);
    }

    /// Wakes every waiting clone except `except`.
    fn notify(&self, except: Option<usize>) {
        let wakers = mem::take(&mut lock(&self.0).wakers);
        wake_all(wakers, except);
    }

    /// Forgets a dropped clone. While a wake of the work is unconsumed, the clone may have been
    /// the one woken to drive it, so the waiting clones take over.
    fn leave(&self, id: usize) {
        let wakers = {
            let mut registry = lock(&self.0);
            registry.wakers.retain(|(entry, _)| *entry != id);
            if registry.woken {
                mem::take(&mut registry.wakers)
            } else {
                Vec::new()
            }
        };
        wake_all(wakers, None);
    }
}

impl Registry {
    fn insert(&mut self, id: &mut Option<usize>, waker: &Waker) {
        let id = *id.get_or_insert_with(|| {
            self.next += 1;
            self.next
        });
        match self.wakers.iter_mut().find(|(entry, _)| *entry == id) {
            Some((_, known)) => known.clone_from(waker),
            None => self.wakers.push((id, waker.clone())),
        }
    }
}

impl Wake for Waiters {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let wakers = {
            let mut registry = lock(&self.0);
            registry.woken = true;
            mem::take(&mut registry.wakers)
        };
        wake_all(wakers, None);
    }
}

fn wake_all(wakers: Vec<(usize, Waker)>, except: Option<usize>) {
    for (id, waker) in wakers {
        if Some(id) != except {
            waker.wake();
        }
    }
}

#[cfg(all(test, loom))]
mod loom;
