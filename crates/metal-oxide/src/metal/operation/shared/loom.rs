//! Loom models of shared work racing with its completing wake and with a dropped clone. Panics stay
//! in the integration tests: an unwinding loom thread poisons the locks of the others.

use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

use loom::{
    sync::atomic::{AtomicBool, Ordering},
    thread,
};

use super::Shared;
use crate::metal::sync::{Mutex, lock};

/// Pending until opened, like a command buffer until it completes; opening wakes the waker of
/// the latest poll.
#[derive(Default)]
struct Gate(Mutex<(bool, Option<Waker>)>);

impl Gate {
    fn open(&self) {
        let waker = {
            let mut state = lock(&self.0);
            state.0 = true;
            state.1.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// Work that completes once its gate is open.
struct Work(Arc<Gate>);

impl Future for Work {
    type Output = u32;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u32> {
        let mut state = lock(&self.0.0);
        if state.0 {
            return Poll::Ready(7);
        }
        state.1 = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[derive(Default)]
struct Flag(AtomicBool);

impl Wake for Flag {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// A clone awaited by its own task, and whether its latest poll completed.
struct Task {
    shared: Shared<Work>,
    flag: Arc<Flag>,
    finished: bool,
}

impl Task {
    fn new(shared: Shared<Work>) -> Self {
        Self {
            shared,
            flag: Arc::default(),
            finished: false,
        }
    }

    fn poll(&mut self) {
        let waker = Waker::from(Arc::clone(&self.flag));
        let polled = Pin::new(&mut self.shared).poll(&mut Context::from_waker(&waker));
        assert!(matches!(polled, Poll::Pending | Poll::Ready(7)));
        self.finished = polled.is_ready();
    }
}

fn work() -> (Arc<Gate>, Shared<Work>) {
    let gate = Arc::new(Gate::default());
    (Arc::clone(&gate), Shared::new(Work(gate)))
}

fn poll_on_thread(shared: Shared<Work>) -> thread::JoinHandle<Task> {
    thread::spawn(move || {
        let mut task = Task::new(shared);
        task.poll();
        task
    })
}

/// Polls woken tasks, as an executor does, until every task finishes. A pending task that
/// nothing woke would wait forever.
fn run(mut tasks: Vec<Task>) {
    loop {
        tasks.retain(|task| !task.finished);
        let woken = tasks
            .iter_mut()
            .find(|task| task.flag.0.swap(false, Ordering::SeqCst));
        match woken {
            Some(task) => task.poll(),
            None => {
                assert!(tasks.is_empty(), "a pending clone was never woken");
                return;
            }
        }
    }
}

#[test]
fn loom_completion_reaches_every_clone_when_only_woken_clones_are_polled() {
    loom::model(|| {
        let (gate, shared) = work();
        let other = poll_on_thread(shared.clone());
        let opener = thread::spawn(move || gate.open());
        let mut task = Task::new(shared);
        task.poll();
        opener.join().unwrap();
        run(vec![task, other.join().unwrap()]);
    });
}

#[test]
fn loom_a_clone_dropped_after_a_wake_hands_the_work_to_the_waiting_clones() {
    loom::model(|| {
        let (gate, shared) = work();
        let waiting = poll_on_thread(shared.clone());
        let opener = thread::spawn(move || gate.open());
        // Its task ends after this poll, even when woken.
        Task::new(shared).poll();
        opener.join().unwrap();
        run(vec![waiting.join().unwrap()]);
    });
}
