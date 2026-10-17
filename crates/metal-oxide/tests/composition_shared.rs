#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

use std::{
    future::{Future, ready},
    pin::Pin,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    thread,
};

use metal_oxide::{Operation, sequence};

struct Token(u32);

#[derive(Debug)]
struct Failure(u32);

struct CountDrop(Arc<AtomicUsize>);

impl Drop for CountDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct NeverReady(CountDrop);

impl Future for NeverReady {
    type Output = Result<Token, Failure>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        let _ = &self.0;
        Poll::Pending
    }
}

/// Pending until opened; then wakes the waker of its latest poll.
#[derive(Default)]
struct Gate(Mutex<(bool, Option<Waker>)>);

impl Gate {
    fn open(&self) {
        let waker = {
            let mut state = self.0.lock().unwrap();
            state.0 = true;
            state.1.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

struct Opened(Arc<Gate>);

impl Future for Opened {
    type Output = Result<Token, Failure>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.0.0.lock().unwrap();
        if state.0 {
            return Poll::Ready(Ok(Token(1)));
        }
        state.1 = Some(cx.waker().clone());
        Poll::Pending
    }
}

#[derive(Default)]
struct Wakes(AtomicUsize);

impl Wake for Wakes {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn poll<F: Future>(future: Pin<&mut F>, wakes: &Arc<Wakes>) -> Poll<F::Output> {
    future.poll(&mut Context::from_waker(&Waker::from(Arc::clone(wakes))))
}

#[test]
fn every_clone_awaits_one_lazy_execution_and_receives_the_same_value() {
    let runs = Arc::new(AtomicUsize::new(0));
    let shared = sequence({
        let runs = Arc::clone(&runs);
        move |_| {
            runs.fetch_add(1, Ordering::SeqCst);
            Ok(Token(7))
        }
    })
    .shared();
    let waiters = (0..4)
        .map(|_| {
            let shared = shared.clone();
            thread::spawn(move || shared.wait().unwrap())
        })
        .collect::<Vec<_>>();
    let values = waiters
        .into_iter()
        .map(|waiter| waiter.join().unwrap())
        .collect::<Vec<_>>();
    let late = shared.clone().wait().unwrap();
    assert_eq!(late.0, 7);
    assert!(values.iter().all(|value| Arc::ptr_eq(value, &late)));
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[test]
fn nothing_runs_until_a_clone_is_polled() {
    let runs = AtomicUsize::new(0);
    let shared = sequence(|_| {
        runs.fetch_add(1, Ordering::SeqCst);
        Ok(Token(2))
    })
    .shared();
    let clones = [shared.clone(), shared.clone()];
    drop(shared);
    drop(clones);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
}

#[test]
fn every_clone_receives_the_same_error_and_skips_its_callbacks() {
    let calls = AtomicUsize::new(0);
    let shared = ready(Err::<Token, _>(Failure(3))).shared();
    let mapped = shared
        .clone()
        .map(|_| calls.fetch_add(1, Ordering::SeqCst))
        .wait()
        .unwrap_err();
    let continued = shared
        .then(|_| {
            calls.fetch_add(1, Ordering::SeqCst);
            ready(Ok(()))
        })
        .wait()
        .unwrap_err();
    assert!(Arc::ptr_eq(&mapped, &continued));
    assert_eq!(mapped.0, 3);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn completion_wakes_the_latest_waker_of_each_remaining_clone() {
    let gate = Arc::new(Gate::default());
    let shared = Opened(Arc::clone(&gate)).shared();
    let mut dropped = Box::pin(shared.clone());
    let mut waiting = Box::pin(shared);
    let [first, stale, latest] = [(); 3].map(|()| Arc::new(Wakes::default()));
    assert!(poll(dropped.as_mut(), &first).is_pending());
    assert!(poll(waiting.as_mut(), &stale).is_pending());
    assert!(poll(waiting.as_mut(), &latest).is_pending());
    drop(dropped);
    gate.open();
    let wakes = [&first, &stale, &latest].map(|wakes| wakes.0.load(Ordering::SeqCst));
    assert_eq!(wakes, [0, 0, 1]);
    let Poll::Ready(Ok(token)) = poll(waiting.as_mut(), &latest) else {
        panic!("the woken clone did not complete");
    };
    assert_eq!(token.0, 1);
}

#[test]
fn dropping_the_last_clone_drops_unfinished_work() {
    let drops = Arc::new(AtomicUsize::new(0));
    let shared = NeverReady(CountDrop(Arc::clone(&drops))).shared();
    let mut polled = Box::pin(shared.clone());
    assert!(poll(polled.as_mut(), &Arc::default()).is_pending());
    drop(polled);
    drop(shared.clone());
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(shared);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}
