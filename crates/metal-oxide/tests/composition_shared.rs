#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

use std::{
    future::{Future, ready},
    panic::{AssertUnwindSafe, catch_unwind},
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

type Hook = Arc<Mutex<Option<Box<dyn FnMut() + Send>>>>;

fn call(hook: &Hook) {
    if let Some(hook) = hook.lock().unwrap().as_mut() {
        hook();
    }
}

/// Wakes during its first poll, as a GPU completion can while that poll runs, then calls
/// `hook` and returns pending; the next poll completes.
struct WakesDuringPoll {
    polls: usize,
    hook: Hook,
}

impl Future for WakesDuringPoll {
    type Output = Result<Token, Failure>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls += 1;
        if self.polls > 1 {
            return Poll::Ready(Ok(Token(5)));
        }
        cx.waker().wake_by_ref();
        call(&self.hook);
        Poll::Pending
    }
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
    assert!(poll(waiting.as_mut(), &latest).is_pending());
    let before = latest.0.load(Ordering::SeqCst);
    gate.open();
    assert!(latest.0.load(Ordering::SeqCst) > before);
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

#[test]
fn dropping_a_woken_clone_wakes_a_clone_that_registered_during_its_poll() {
    let hook = Hook::default();
    let shared = WakesDuringPoll {
        polls: 0,
        hook: Arc::clone(&hook),
    }
    .shared();
    let mut driver = Box::pin(shared.clone());
    let waiter = Arc::new(Mutex::new(Box::pin(shared)));
    let waited = Arc::new(Wakes::default());
    *hook.lock().unwrap() = Some(Box::new({
        let (waiter, waited) = (Arc::clone(&waiter), Arc::clone(&waited));
        move || assert!(poll(waiter.lock().unwrap().as_mut(), &waited).is_pending())
    }));
    assert!(poll(driver.as_mut(), &Arc::default()).is_pending());
    drop(driver);
    assert!(waited.0.load(Ordering::SeqCst) > 0);
    let Poll::Ready(Ok(token)) = poll(waiter.lock().unwrap().as_mut(), &waited) else {
        panic!("the woken clone did not complete");
    };
    assert_eq!(token.0, 5);
}

#[test]
fn a_panic_in_shared_work_reaches_a_clone_that_registered_during_it() {
    let hook = Hook::default();
    let shared = ready(Ok::<_, Failure>(1_u32))
        .map({
            let hook = Arc::clone(&hook);
            move |_| -> u32 {
                call(&hook);
                panic!("the shared callback failed")
            }
        })
        .shared();
    let mut driver = Box::pin(shared.clone());
    let waiter = Arc::new(Mutex::new(Box::pin(shared)));
    let waited = Arc::new(Wakes::default());
    *hook.lock().unwrap() = Some(Box::new({
        let (waiter, waited) = (Arc::clone(&waiter), Arc::clone(&waited));
        move || assert!(poll(waiter.lock().unwrap().as_mut(), &waited).is_pending())
    }));
    assert!(catch_unwind(AssertUnwindSafe(|| poll(driver.as_mut(), &Arc::default()))).is_err());
    drop(driver);
    assert!(waited.0.load(Ordering::SeqCst) > 0);
    let polled = catch_unwind(AssertUnwindSafe(|| {
        poll(waiter.lock().unwrap().as_mut(), &waited).is_ready()
    }));
    assert!(polled.is_err(), "the clone did not observe the panic");
}
