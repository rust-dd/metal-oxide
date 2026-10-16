//! Loom models of the runtime's cross-thread state. Run with
//! `RUSTFLAGS="--cfg loom" cargo test -p metal-oxide --lib --release loom`.

use std::{
    cell::UnsafeCell,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Wake, Waker},
};

use loom::{sync::atomic::AtomicBool, thread};
use metal_oxide_artifact::{Access, Layout};

use super::{
    access::AccessState,
    buffer::{Location, Mirror},
    classic::capacity::{Capacity, LIMIT},
    completion::Completion,
};
use crate::GpuValue;

/// Decode counts per model; tests run concurrently, so each model owns one counter.
static DECODES: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

#[derive(Clone, Copy, Debug, PartialEq)]
struct Counted<const MODEL: usize>(u32);

impl<const MODEL: usize> GpuValue for Counted<MODEL> {
    const SIZE: usize = 4;
    const ALIGNMENT: usize = 4;
    fn layout() -> crate::Result<Layout> {
        u32::layout()
    }
    fn zeroed() -> Self {
        Self(0)
    }
    fn encode(self, bytes: &mut [u8]) {
        self.0.encode(bytes);
    }
    fn decode(bytes: &[u8]) -> Self {
        DECODES[MODEL].fetch_add(1, Ordering::Relaxed);
        Self(u32::decode(bytes))
    }
}

/// Canonical bytes standing in for a buffer's shared Metal storage.
struct Storage(UnsafeCell<Vec<u8>>);

// SAFETY: the mirror protocol under test serializes every access to the bytes.
unsafe impl Sync for Storage {}

impl Storage {
    fn new(values: &[u32]) -> Arc<Self> {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        Arc::new(Self(UnsafeCell::new(bytes)))
    }

    fn bytes(&self) -> NonNull<u8> {
        // SAFETY: the vector is never resized; the pointer covers its initialized bytes.
        NonNull::new(unsafe { (*self.0.get()).as_mut_ptr() }).unwrap()
    }
}

struct Flag(AtomicBool);

impl Wake for Flag {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn finish(completion: &Arc<Completion>) -> thread::JoinHandle<()> {
    let completion = Arc::clone(completion);
    thread::spawn(move || completion.finish(Ok(None)))
}

#[test]
fn loom_completion_wakes_a_pending_poll_and_releases_waiters() {
    loom::model(|| {
        let completion = Arc::new(Completion::default());
        let finisher = finish(&completion);
        let flag = Arc::new(Flag(AtomicBool::new(false)));
        let waker = Waker::from(Arc::clone(&flag));
        let polled = completion.poll(&mut Context::from_waker(&waker));
        assert!(completion.wait().is_ok());
        finisher.join().unwrap();
        assert!(polled.is_ready() || flag.0.load(Ordering::SeqCst));
    });
}

#[test]
fn loom_completion_wakes_every_task_polling_it() {
    loom::model(|| {
        let completion = Arc::new(Completion::default());
        let finisher = finish(&completion);
        let polls = [0, 1].map(|_| {
            let flag = Arc::new(Flag(AtomicBool::new(false)));
            let waker = Waker::from(Arc::clone(&flag));
            let polled = completion.poll(&mut Context::from_waker(&waker));
            (polled.is_ready(), flag)
        });
        let reader = {
            let completion = Arc::clone(&completion);
            thread::spawn(move || {
                let flag = Arc::new(Flag(AtomicBool::new(false)));
                let waker = Waker::from(Arc::clone(&flag));
                let polled = completion.poll_finished(&mut Context::from_waker(&waker));
                (polled.is_ready(), flag)
            })
        };
        let reader = reader.join().unwrap();
        // Wakers run after the result is published, so they are checked once `finish` returns.
        finisher.join().unwrap();
        for (ready, flag) in polls.into_iter().chain([reader]) {
            assert!(ready || flag.0.load(Ordering::SeqCst));
        }
    });
}

#[test]
fn loom_polled_readback_is_ready_only_after_a_registered_write() {
    loom::model(|| {
        let access = Arc::new(AccessState::default());
        let write = Arc::new(Completion::default());
        access.register(super::access::write(&write));
        let finisher = finish(&write);
        let flag = Arc::new(Flag(AtomicBool::new(false)));
        let waker = Waker::from(Arc::clone(&flag));
        let mut cx = Context::from_waker(&waker);
        if access.poll(false, &mut cx).is_ready() {
            assert!(write.is_finished());
        }
        finisher.join().unwrap();
        assert!(flag.0.load(Ordering::SeqCst) || access.poll(false, &mut cx).is_ready());
        assert!(access.poll(false, &mut cx).is_ready());
        assert!(access.is_idle());
    });
}

#[test]
fn loom_concurrent_readers_wait_for_a_registered_write() {
    loom::model(|| {
        let access = Arc::new(AccessState::default());
        let write = Arc::new(Completion::default());
        access.register(super::access::write(&write));
        let finisher = finish(&write);
        let reader = {
            let (access, write) = (Arc::clone(&access), Arc::clone(&write));
            thread::spawn(move || {
                access.synchronize(false);
                assert!(write.is_finished());
            })
        };
        access.synchronize(false);
        assert!(write.is_finished());
        reader.join().unwrap();
        finisher.join().unwrap();
        assert!(access.is_idle());
    });
}

#[test]
fn loom_recording_commit_never_exposes_an_idle_buffer() {
    loom::model(|| {
        let access = Arc::new(AccessState::default());
        let mut reservation = access.reserve(true).unwrap();
        let completion = Arc::new(Completion::default());
        let committer = {
            let completion = Arc::clone(&completion);
            thread::spawn(move || {
                reservation.register(super::access::write(&completion));
                drop(reservation);
                thread::yield_now();
                completion.finish(Ok(None));
            })
        };
        let polled = catch_unwind(AssertUnwindSafe(|| {
            access.poll(false, &mut Context::from_waker(Waker::noop()))
        }));
        if polled.is_ok_and(|poll| poll.is_ready()) {
            assert!(completion.is_finished());
        }
        committer.join().unwrap();
        access.synchronize(false);
        assert!(access.take_written());
        assert!(access.is_idle());
    });
}

#[test]
fn loom_conflicting_recordings_never_overlap() {
    loom::model(|| {
        let access = Arc::new(AccessState::default());
        let active = Arc::new(AtomicBool::new(false));
        let workers = [0, 1].map(|_| {
            let (access, active) = (Arc::clone(&access), Arc::clone(&active));
            thread::spawn(move || {
                if let Ok(_reservation) = access.reserve(true) {
                    assert!(!active.swap(true, Ordering::SeqCst));
                    thread::yield_now();
                    active.store(false, Ordering::SeqCst);
                }
            })
        });
        for worker in workers {
            worker.join().unwrap();
        }
        assert!(access.is_idle());
    });
}

#[test]
fn loom_shared_recordings_exclude_write_upgrades_until_released() {
    loom::model(|| {
        let access = Arc::new(AccessState::default());
        let reservation = access.reserve(false).unwrap();
        let reader = {
            let access = Arc::clone(&access);
            thread::spawn(move || {
                let other = access.reserve(false).unwrap();
                access.synchronize(false);
                assert!(
                    access
                        .poll(false, &mut Context::from_waker(Waker::noop()))
                        .is_ready()
                );
                assert!(other.upgrade(true).is_err());
            })
        };
        reader.join().unwrap();
        reservation.upgrade(true).unwrap().keep();
        assert!(access.reserve(false).is_err());
        drop(reservation);
        assert!(access.is_idle());
    });
}

#[test]
fn loom_registration_excludes_other_batches_until_native_submission() {
    loom::model(|| {
        let access = Arc::new(AccessState::default());
        let mut reservation = access.reserve(true).unwrap();
        let completion = Arc::new(Completion::default());
        let submitted = Arc::new(AtomicBool::new(false));
        let writer = {
            let submitted = Arc::clone(&submitted);
            thread::spawn(move || {
                reservation.register(super::access::write(&completion));
                thread::yield_now();
                submitted.store(true, Ordering::SeqCst);
                drop(reservation);
                completion.finish(Ok(None));
            })
        };
        if let Ok(_reader) = access.reserve(false) {
            assert!(submitted.load(Ordering::SeqCst));
        }
        writer.join().unwrap();
        access.synchronize(false);
        assert!(access.is_idle());
    });
}

#[test]
fn loom_capacity_release_wakes_a_racing_waiter() {
    loom::model(|| {
        let capacity = Arc::new(Capacity::default());
        let mut permits = (0..LIMIT)
            .map(|_| {
                let std::task::Poll::Ready(permit) =
                    capacity.poll(&mut Context::from_waker(Waker::noop()))
                else {
                    panic!("expected available command storage");
                };
                permit
            })
            .collect::<Vec<_>>();
        let permit = permits.pop().unwrap();
        let releaser = thread::spawn(move || drop(permit));
        let flag = Arc::new(Flag(AtomicBool::new(false)));
        let waker = Waker::from(Arc::clone(&flag));
        let polled = capacity.poll(&mut Context::from_waker(&waker));
        releaser.join().unwrap();
        assert!(polled.is_ready() || flag.0.load(Ordering::SeqCst));
    });
}

#[test]
fn loom_concurrent_readbacks_decode_a_gpu_write_once() {
    loom::model(|| {
        DECODES[0].store(0, Ordering::Relaxed);
        let access = Arc::new(AccessState::default());
        let mirror = Arc::new(Mirror::new(vec![Counted::<0>(0); 2], Location::Synced));
        let storage = Storage::new(&[7, 9]);
        access.mark_written();
        let reader = {
            let (access, mirror, storage) = (
                Arc::clone(&access),
                Arc::clone(&mirror),
                Arc::clone(&storage),
            );
            // SAFETY: the storage holds two canonical values and no GPU access is pending.
            thread::spawn(move || unsafe { mirror.download(&access, storage.bytes()) }.to_vec())
        };
        // SAFETY: as above.
        let values = unsafe { mirror.download(&access, storage.bytes()) }.to_vec();
        assert_eq!(values, [Counted(7), Counted(9)]);
        assert_eq!(reader.join().unwrap(), values);
        assert_eq!(DECODES[0].load(Ordering::Relaxed), 2);
    });
}

#[test]
fn loom_readback_and_encoding_agree_on_cpu_written_values() {
    loom::model(|| {
        DECODES[1].store(0, Ordering::Relaxed);
        let access = Arc::new(AccessState::default());
        let mirror = Arc::new(Mirror::new(
            vec![Counted::<1>(3), Counted(4)],
            Location::Cpu,
        ));
        let storage = Storage::new(&[0, 0]);
        let encoder = {
            let (access, mirror, storage) = (
                Arc::clone(&access),
                Arc::clone(&mirror),
                Arc::clone(&storage),
            );
            thread::spawn(move || {
                let _reservation = access.reserve(false).unwrap();
                // SAFETY: CPU-modified values guarantee that no GPU access is pending.
                unsafe { mirror.prepare(Access::Read, storage.bytes()) }
            })
        };
        access.synchronize(false);
        // SAFETY: the storage holds two canonical values and no GPU access is pending.
        let values = unsafe { mirror.download(&access, storage.bytes()) }.to_vec();
        encoder.join().unwrap();
        assert_eq!(values, [Counted(3), Counted(4)]);
        // SAFETY: both threads finished; the bytes are no longer shared.
        assert_eq!(unsafe { &*storage.0.get() }, &[3, 0, 0, 0, 4, 0, 0, 0]);
        assert_eq!(DECODES[1].load(Ordering::Relaxed), 0);
    });
}
