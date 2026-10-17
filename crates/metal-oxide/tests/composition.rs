#![cfg(all(target_os = "macos", target_arch = "aarch64", not(loom)))]

use std::{
    future::{Future, ready},
    pin::{Pin, pin},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll, Waker},
};

use metal_oxide::{Operation, sequence};

struct CountDrop(Arc<AtomicUsize>);

impl Drop for CountDrop {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

struct NeverReady(CountDrop);

impl Future for NeverReady {
    type Output = Result<usize, &'static str>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        let _ = &self.0;
        Poll::Pending
    }
}

fn pending<F: Future>(future: Pin<&mut F>) {
    assert!(
        future
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
}

#[test]
fn map_and_then_are_lazy_and_pass_owned_values_after_success() {
    let calls = AtomicUsize::new(0);
    let operation = sequence(|_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(String::from("kernel"))
    })
    .map(|mut value| {
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        calls.fetch_add(1, Ordering::SeqCst);
        value.push_str(" output");
        value
    })
    .then(|value| {
        let calls = &calls;
        async move {
            assert_eq!(calls.load(Ordering::SeqCst), 2);
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(value.len())
        }
    });
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(operation.wait().unwrap(), 13);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn failures_preserve_the_error_and_skip_every_callback() {
    let calls = AtomicUsize::new(0);
    let error = Box::new(37_u32);
    let address = &*error as *const u32;
    let result = ready(Err::<String, _>(error))
        .map(|value| {
            calls.fetch_add(1, Ordering::SeqCst);
            value.len()
        })
        .then(|value| {
            calls.fetch_add(1, Ordering::SeqCst);
            ready(Ok(value + 1))
        })
        .wait();
    let error = result.unwrap_err();
    assert_eq!(&*error as *const u32, address);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn an_unpolled_chain_drops_its_captures_without_starting() {
    let drops = Arc::new(AtomicUsize::new(0));
    let captured = CountDrop(Arc::clone(&drops));
    let calls = AtomicUsize::new(0);
    let operation = sequence(|_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(captured)
    })
    .map(drop);
    drop(operation);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn dropping_the_first_pending_stage_releases_future_and_callback_captures() {
    let drops = Arc::new(AtomicUsize::new(0));
    let captured = CountDrop(Arc::clone(&drops));
    let calls = AtomicUsize::new(0);
    {
        let operation = NeverReady(CountDrop(Arc::clone(&drops))).then(|value| {
            let calls = &calls;
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                drop(captured);
                Ok(value)
            }
        });
        let mut future = pin!(operation);
        pending(future.as_mut());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn dropping_the_second_pending_stage_does_not_run_later_callbacks() {
    let drops = Arc::new(AtomicUsize::new(0));
    let calls = AtomicUsize::new(0);
    {
        let operation = ready(Ok::<_, &'static str>(CountDrop(Arc::clone(&drops))))
            .then(|value| {
                calls.fetch_add(1, Ordering::SeqCst);
                NeverReady(value)
            })
            .map(|_| panic!("the second stage never completes"));
        let mut future = pin!(operation);
        pending(future.as_mut());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn timed_batch_results_can_be_processed_after_completion() {
    let output = sequence(|_| Ok(41))
        .timed()
        .map(|(value, duration)| {
            assert!(duration.is_none());
            value + 1
        })
        .wait()
        .unwrap();
    assert_eq!(output, 42);
}

#[test]
fn composed_futures_move_between_executor_threads() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .build()
        .unwrap();
    runtime.block_on(async {
        let operation = sequence(|_| Ok(Box::new(6_u32)))
            .then(|value| async move {
                tokio::task::yield_now().await;
                Ok(*value * 7)
            })
            .map(|value| value.to_string());
        assert_eq!(tokio::spawn(operation).await.unwrap().unwrap(), "42");
    });
}
