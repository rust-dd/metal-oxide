use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Wake;

use super::*;

#[derive(Default)]
struct Count(AtomicUsize);

impl Wake for Count {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

fn reserve(capacity: &Arc<Capacity>) -> Permit {
    let Poll::Ready(permit) = capacity.poll(&mut Context::from_waker(Waker::noop())) else {
        panic!("expected available command storage");
    };
    permit
}

#[test]
fn saturated_capacity_wakes_each_waiter_after_an_encoder_is_cancelled() {
    let capacity = Arc::new(Capacity::default());
    let mut permits = (0..LIMIT).map(|_| reserve(&capacity)).collect::<Vec<_>>();
    let first = Arc::new(Count::default());
    let second = Arc::new(Count::default());
    for flag in [&first, &first, &second] {
        let waker = Waker::from(Arc::clone(flag));
        assert!(capacity.poll(&mut Context::from_waker(&waker)).is_pending());
    }
    drop(permits.pop());
    assert_eq!(first.0.load(Ordering::Relaxed), 1);
    assert_eq!(second.0.load(Ordering::Relaxed), 1);
    permits.push(reserve(&capacity));
    assert!(
        capacity
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    drop(permits);
    assert_eq!(lock(&capacity.0).used, 0);
}

#[test]
fn dropping_a_completed_guard_returns_only_its_own_slot() {
    let capacity = Arc::new(Capacity::default());
    let first = reserve(&capacity);
    let second = reserve(&capacity);
    drop(first);
    assert_eq!(lock(&capacity.0).used, 1);
    drop(second);
    assert_eq!(lock(&capacity.0).used, 0);
}

#[test]
fn a_panicking_waker_does_not_prevent_other_waiters_from_waking() {
    struct Panics;
    impl Wake for Panics {
        fn wake(self: Arc<Self>) {
            panic!("waker panicked");
        }
    }
    let capacity = Arc::new(Capacity::default());
    let permits = (0..LIMIT).map(|_| reserve(&capacity)).collect::<Vec<_>>();
    let panic_waker = Waker::from(Arc::new(Panics));
    assert!(
        capacity
            .poll(&mut Context::from_waker(&panic_waker))
            .is_pending()
    );
    let count = Arc::new(Count::default());
    let waker = Waker::from(Arc::clone(&count));
    assert!(capacity.poll(&mut Context::from_waker(&waker)).is_pending());
    drop(permits);
    assert_eq!(count.0.load(Ordering::Relaxed), 1);
    assert_eq!(lock(&capacity.0).used, 0);
}
