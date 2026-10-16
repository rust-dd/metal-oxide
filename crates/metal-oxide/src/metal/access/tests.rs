use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    task::Waker,
};

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
    access.synchronize(false);
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
    assert!(access.poll(false, &mut cx).is_pending());
    first.finish(Ok(None));
    assert!(access.poll(false, &mut cx).is_pending());
    second.finish(Ok(None));
    assert!(access.poll(false, &mut cx).is_ready());
    assert!(access.is_idle());
}

#[test]
fn recording_reservations_reject_conflicting_cpu_access_and_release_on_drop() {
    for gpu_write in [false, true] {
        let access = Arc::new(AccessState::default());
        let reservation = access.reserve(gpu_write).unwrap();
        for cpu_write in [false, true] {
            assert_eq!(
                catch_unwind(AssertUnwindSafe(|| access.synchronize(cpu_write))).is_err(),
                gpu_write || cpu_write,
            );
            assert_eq!(
                catch_unwind(AssertUnwindSafe(|| {
                    access.poll(cpu_write, &mut Context::from_waker(Waker::noop()))
                }))
                .is_err(),
                gpu_write || cpu_write,
            );
        }
        assert!(!access.is_idle());
        drop(reservation);
        access.synchronize(true);
        assert!(access.is_idle());
    }
}

#[test]
fn read_reservations_share_and_write_upgrades_are_transactional() {
    let access = Arc::new(AccessState::default());
    let mut first = access.reserve(false).unwrap();
    let second = access.reserve(false).unwrap();
    assert!(access.reserve(true).is_err());
    assert!(first.upgrade(true).is_err());
    drop(second);
    let upgrade = first.upgrade(true).unwrap();
    assert!(access.reserve(false).is_err());
    drop(upgrade);
    drop(access.reserve(false).unwrap());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _upgrade = first.upgrade(true).unwrap();
            panic!("aborted upgrade");
        }))
        .is_err()
    );
    drop(access.reserve(false).unwrap());
    first.upgrade(true).unwrap().keep();
    assert!(access.reserve(false).is_err());
    let completion = Arc::new(Completion::default());
    first.register(write(&completion));
    assert!(access.reserve(false).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| access.assert_unrecorded())).is_err());
    drop(first);
    assert!(
        access
            .poll(false, &mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    completion.finish(Ok(None));
    access.synchronize(false);
    assert!(access.take_written());
    assert!(access.is_idle());
}

#[test]
fn forgetting_a_reservation_keeps_cpu_access_blocked() {
    let access = Arc::new(AccessState::default());
    std::mem::forget(access.reserve(true).unwrap());
    assert!(access.reserve(false).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| access.assert_unrecorded())).is_err());
}
