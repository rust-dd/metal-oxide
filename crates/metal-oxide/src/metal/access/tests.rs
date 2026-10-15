use std::task::Waker;

use super::*;

#[test]
fn buffer_access_waits_for_every_pending_command_buffer() {
    let access = AccessState::default();
    let first = Arc::new(Completion::default());
    let second = Arc::new(Completion::default());
    access.register(write(&first));
    access.register(write(&second));
    let worker = std::thread::spawn(move || {
        first.finish(Ok(None));
        second.finish(Ok(None));
    });
    access.synchronize();
    worker.join().unwrap();
    assert!(access.is_idle());
}

#[test]
fn polled_access_is_ready_only_after_every_registered_completion() {
    let access = AccessState::default();
    let first = Arc::new(Completion::default());
    let second = Arc::new(Completion::default());
    access.register(write(&first));
    access.register(write(&second));
    let mut cx = Context::from_waker(Waker::noop());
    assert!(access.poll(&mut cx).is_pending());
    first.finish(Ok(None));
    assert!(access.poll(&mut cx).is_pending());
    second.finish(Ok(None));
    assert!(access.poll(&mut cx).is_ready());
    assert!(access.is_idle());
}
