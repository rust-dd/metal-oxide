#[cfg(not(loom))]
mod native {
    use std::{
        sync::{Arc, mpsc},
        task::{Context, Wake, Waker},
        time::Duration,
    };

    use super::super::Preparation;
    use crate::Error;

    struct Notify(mpsc::Sender<()>);

    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            let _ = self.0.send(());
        }
    }

    fn notification() -> (Waker, mpsc::Receiver<()>) {
        let (send, receive) = mpsc::channel();
        (Waker::from(Arc::new(Notify(send))), receive)
    }

    #[test]
    fn blocked_preparation_returns_pending_and_wakes_the_latest_poller() {
        let (started, running) = mpsc::channel();
        let (release, wait) = mpsc::channel();
        let preparation = Preparation::new(move || {
            let _ = started.send(());
            let _ = wait.recv();
            Err(Error::KernelNotFound("missing".into()))
        });
        running.recv_timeout(Duration::from_secs(5)).unwrap();
        let (first, first_wake) = notification();
        let (last, last_wake) = notification();
        assert!(
            preparation
                .poll(&mut Context::from_waker(&first))
                .is_pending()
        );
        assert!(
            preparation
                .poll(&mut Context::from_waker(&last))
                .is_pending()
        );
        release.send(()).unwrap();
        last_wake.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(first_wake.try_recv().is_err());
        assert!(matches!(
            preparation.poll(&mut Context::from_waker(&last)),
            std::task::Poll::Ready(Err(Error::KernelNotFound(name))) if name == "missing"
        ));
    }

    #[test]
    fn completed_preparation_delivers_its_error_on_the_first_poll() {
        let preparation = Preparation(Arc::new(super::super::Mutex::new(super::super::State {
            result: Some(Err(Error::DeviceMismatch)),
            waker: None,
        })));
        assert!(matches!(
            preparation.poll(&mut Context::from_waker(Waker::noop())),
            std::task::Poll::Ready(Err(Error::DeviceMismatch))
        ));
    }

    #[test]
    fn panicking_waker_cannot_unwind_from_worker_completion() {
        struct Panic;
        impl Wake for Panic {
            fn wake(self: Arc<Self>) {
                panic!("waker panic");
            }
        }
        let preparation = Preparation(Arc::new(super::super::Mutex::new(super::super::State {
            result: None,
            waker: Some(Waker::from(Arc::new(Panic))),
        })));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                preparation.complete(Err(Error::DeviceMismatch));
            }))
            .is_ok()
        );
        assert!(matches!(
            preparation.poll(&mut Context::from_waker(Waker::noop())),
            std::task::Poll::Ready(Err(Error::DeviceMismatch))
        ));
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn ready_pipeline_needs_no_preparation() {
        let device = crate::Device::system_default().unwrap();
        let module =
            crate::Module::from_source(&device, "#include <metal_stdlib>\nkernel void empty() {}")
                .unwrap();
        let pipeline = crate::Pipeline::new(&device, &module, "empty").unwrap();
        let mut source = super::super::Source::ready(pipeline);
        assert!(matches!(source, super::super::Source::Ready(_)));
        assert!(matches!(
            source.poll(&mut Context::from_waker(Waker::noop())),
            std::task::Poll::Ready(Ok(()))
        ));
        assert_eq!(source.device().id(), device.id());
        assert!(source.pipeline().thread_execution_width() > 0);
    }

    #[test]
    fn dropping_preparation_does_not_wait_for_its_worker() {
        let (release, wait) = mpsc::channel();
        let (completed, receive) = mpsc::channel();
        let (dropped, drop_finished) = mpsc::channel();
        let preparation = Preparation::new(move || {
            let _ = wait.recv();
            let _ = completed.send(());
            Err(Error::DeviceMismatch)
        });
        let dropping = std::thread::spawn(move || {
            drop(preparation);
            let _ = dropped.send(());
        });
        drop_finished.recv_timeout(Duration::from_secs(5)).unwrap();
        release.send(()).unwrap();
        receive.recv_timeout(Duration::from_secs(5)).unwrap();
        dropping.join().unwrap();
    }
}

#[cfg(loom)]
mod models {
    use std::{
        sync::Arc,
        task::{Context, Poll, Wake, Waker},
    };

    use loom::sync::atomic::{AtomicBool, Ordering};

    use super::super::{Mutex, Preparation, State};
    use crate::Error;

    struct Notify(AtomicBool);

    impl Wake for Notify {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::Release);
        }
    }

    #[test]
    fn loom_preparation_completion_cannot_lose_a_wake() {
        loom::model(|| {
            let preparation = Arc::new(Preparation(Arc::new(Mutex::new(State {
                result: None,
                waker: None,
            }))));
            let notify = Arc::new(Notify(AtomicBool::new(false)));
            let waker = Waker::from(Arc::clone(&notify));
            let worker = Arc::clone(&preparation);
            let job = loom::thread::spawn(move || worker.complete(Err(Error::DeviceMismatch)));
            let result = preparation.poll(&mut Context::from_waker(&waker));
            job.join().unwrap();
            if result.is_pending() {
                assert!(notify.0.load(Ordering::Acquire));
                assert!(matches!(
                    preparation.poll(&mut Context::from_waker(&waker)),
                    Poll::Ready(Err(Error::DeviceMismatch))
                ));
            } else {
                assert!(matches!(result, Poll::Ready(Err(Error::DeviceMismatch))));
            }
        });
    }
}
