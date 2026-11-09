//! Loom model of classic completion-event reuse: a completion handler returns its command
//! buffer's event to the pool, and the next command buffer takes it with the next value.

use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

use loom::thread;
use objc2_metal::MTLSharedEvent;

use super::{Finish, Signal, Signals};
use crate::metal::{loom::device, sync::lock};

/// A committed classic command buffer: its work and the event value set when the work ends.
struct Committed {
    /// A std atomic: native events give loom no ordering, so it must observe the same execution.
    done: Arc<AtomicBool>,
    signal: Signal,
    handler: Finish,
}

fn commit(signals: &Arc<Signals>) -> Committed {
    let signal = signals.take(device()).unwrap();
    let returned = Some((signal.clone(), Arc::clone(signals)));
    Committed {
        done: Arc::new(AtomicBool::new(false)),
        handler: Finish::new(Arc::new(AtomicUsize::new(1)), returned),
        signal,
    }
}

impl Committed {
    /// The GPU ends the work and sets the event; then the completion handler runs.
    fn finish(self) {
        self.done.store(true, Ordering::SeqCst);
        self.signal.event.setSignaledValue(self.signal.value);
        self.handler.run();
    }

    /// The command buffer fails, leaving its event unset; then the completion handler runs.
    fn fail(self) {
        self.done.store(true, Ordering::SeqCst);
        self.handler.run();
    }

    /// A command buffer on another queue that waits for the event before reading the results.
    fn waiter(&self) -> thread::JoinHandle<()> {
        let (signal, done) = (self.signal.clone(), Arc::clone(&self.done));
        thread::spawn(move || {
            while signal.event.signaledValue() < signal.value {
                thread::yield_now();
            }
            assert!(
                done.load(Ordering::SeqCst),
                "a waiter started before the work ended"
            );
        })
    }
}

#[test]
fn loom_a_reused_event_releases_only_the_waiters_of_finished_work() {
    loom::model(|| {
        let signals = Arc::new(Signals::default());
        let first = commit(&signals);
        let first_waiter = first.waiter();
        let first_gpu = thread::spawn(move || first.fail());
        let second = commit(&signals);
        let second_waiter = second.waiter();
        let second_gpu = thread::spawn(move || second.finish());
        for thread in [first_waiter, first_gpu, second_waiter, second_gpu] {
            thread.join().unwrap();
        }
        for signal in lock(&signals.0).iter() {
            assert_eq!(signal.event.signaledValue(), signal.value);
        }
    });
}
