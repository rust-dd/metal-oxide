use std::{sync::mpsc, task::Waker, time::Duration};

use objc2_metal::MTLCreateSystemDefaultDevice;

use super::*;

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn cancelled_encoders_release_native_storage_before_returning_capacity() {
    let (sender, receiver) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        autoreleasepool(|_| {
            let device = MTLCreateSystemDefaultDevice().unwrap();
            let queue = ClassicQueue::new(&device).unwrap();
            let mut cx = Context::from_waker(Waker::noop());
            let mut batches = (0..LIMIT)
                .map(|_| {
                    let Poll::Ready(batch) =
                        ClassicBatch::poll_new(&queue, None, true, &[], &mut cx)
                    else {
                        panic!("native queue must hold its configured capacity");
                    };
                    batch.unwrap()
                })
                .collect::<Vec<_>>();
            assert!(ClassicBatch::poll_new(&queue, None, true, &[], &mut cx).is_pending());
            for _ in 0..LIMIT * 2 {
                drop(batches.pop());
                let Poll::Ready(batch) = ClassicBatch::poll_new(&queue, None, true, &[], &mut cx)
                else {
                    panic!("cancelled encoder must return its capacity");
                };
                batches.push(batch.unwrap());
            }
        });
        sender.send(()).unwrap();
    });
    receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("native storage allocation blocked after an encoder was cancelled");
    worker.join().unwrap();
}
