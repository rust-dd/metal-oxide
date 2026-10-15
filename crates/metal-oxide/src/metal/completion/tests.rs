use super::*;
use objc2_foundation::{NSError, NSString};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::task::Wake;

struct Counter(AtomicUsize);
impl Wake for Counter {
    fn wake(self: Arc<Self>) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
#[cfg_attr(miri, ignore = "creates an NSError through Objective-C")]
fn terminal_memory_errors_survive_wait_and_poll() {
    for (domain, code, memory) in [
        ("MTLCommandBufferErrorDomain", 8, true),
        ("MTL4CommandQueueErrorDomain", 3, true),
        ("MTLCommandBufferErrorDomain", 3, false),
        ("MTL4CommandQueueErrorDomain", 8, false),
        ("unrelated", 8, false),
        ("unrelated", 3, false),
    ] {
        // SAFETY: no user-info dictionary or untyped objects are supplied.
        let error = unsafe {
            NSError::errorWithDomain_code_userInfo(&NSString::from_str(domain), code, None)
        };
        let completion = Completion::default();
        completion.finish(Err(CommandFailure::from_native(&error)));
        for _ in 0..2 {
            let error = completion.wait().unwrap_err();
            assert_eq!(matches!(error, Error::OutOfMemory(_)), memory, "{error}");
            assert!(error.to_string().contains(domain));
            let Poll::Ready(Err(error)) = completion.poll(&mut Context::from_waker(Waker::noop()))
            else {
                panic!("terminal failure was lost");
            };
            assert_eq!(matches!(error, Error::OutOfMemory(_)), memory, "{error}");
        }
    }
}

#[test]
fn native_timestamps_reject_unavailable_and_invalid_measurements() {
    assert_eq!(gpu_time(12.0, 12.25), Some(Duration::from_millis(250)));
    for (start, end) in [
        (0.0, 0.0),
        (2.0, 1.0),
        (-1.0, 1.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
    ] {
        assert_eq!(gpu_time(start, end), None);
    }
}

#[test]
fn completion_wakes_every_distinct_waiter_once_and_preserves_errors() {
    let completion = Completion::default();
    let first = Arc::new(Counter(AtomicUsize::new(0)));
    let second = Arc::new(Counter(AtomicUsize::new(0)));
    for counter in [&first, &second, &first] {
        let waker = Waker::from(Arc::clone(counter));
        assert!(
            completion
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
    }
    let third = Arc::new(Counter(AtomicUsize::new(0)));
    let waker = Waker::from(Arc::clone(&third));
    assert!(
        completion
            .poll_finished(&mut Context::from_waker(&waker))
            .is_pending()
    );
    completion.finish(Err(CommandFailure::Other("test command failure".into())));
    for counter in [&first, &second, &third] {
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
    }
    assert_eq!(
        completion.wait().unwrap_err().to_string(),
        "Metal command failed: test command failure"
    );
    assert!(
        completion
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_ready()
    );
    assert!(
        completion
            .poll_finished(&mut Context::from_waker(Waker::noop()))
            .is_ready()
    );
}
