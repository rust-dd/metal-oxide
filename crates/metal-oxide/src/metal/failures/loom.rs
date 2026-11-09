//! Loom models of failure propagation down a chain of command buffers. The GPU work of the chain
//! has completed; the completion handlers run concurrently, in any order.

use std::sync::Arc;

use loom::thread;
use objc2_metal::MTLBuffer;

use super::{Failed, FailurePool, Failures, KernelChecks, Slot};
use crate::{
    KernelFailure,
    metal::{loom::device, sync::lock},
};

fn kernel() -> Arc<KernelChecks> {
    Arc::new(KernelChecks {
        name: "fallible".into(),
        gate_binding: 0,
        fallible: true,
        lengths: false,
        checks: Arc::from([]),
    })
}

/// A command buffer with one fallible dispatch, which failed when `failed` holds.
fn checked(pool: &Arc<FailurePool>, upstream: Option<&Arc<Failures>>, failed: bool) -> Failures {
    let mut failures = Failures::new(Arc::clone(pool));
    if let Some(upstream) = upstream {
        failures
            .follow(device(), vec![Arc::clone(upstream)])
            .unwrap();
    }
    let word = failures.failure_word(device(), &kernel(), 0).unwrap();
    if failed {
        fail(&failures, word);
    }
    failures
}

/// Sets a failure word as a failing GPU thread does, to its check's index plus two.
fn fail(failures: &Failures, word: Slot) {
    let bytes = failures.buffer(word).contents().cast::<u8>();
    // SAFETY: the word is inside its chunk, and no GPU work uses the chunk.
    unsafe { bytes.add(word.offset).cast::<u32>().write(2) };
}

/// Three command buffers, each reading the writes of the one before; `failed` picks the ones
/// whose dispatch failed.
fn chain(pool: &Arc<FailurePool>, failed: [bool; 3]) -> [Arc<Failures>; 3] {
    let first = Arc::new(checked(pool, None, failed[0]));
    let second = Arc::new(checked(pool, Some(&first), failed[1]));
    let third = Arc::new(checked(pool, Some(&second), failed[2]));
    [first, second, third]
}

/// Finishes the chain on three threads; returns what each command buffer reported.
fn finish_concurrently([first, second, third]: [Arc<Failures>; 3]) -> [Option<Failed>; 3] {
    let handlers = [first, third].map(|failures| thread::spawn(move || failures.finish()));
    let middle = second.finish();
    let [first, third] = handlers.map(|handler| handler.join().unwrap());
    [first, middle, third]
}

fn dependency(failed: Option<Failed>) -> Option<KernelFailure> {
    match failed {
        Some(Failed::Dependency(failure)) => Some(failure),
        _ => None,
    }
}

#[test]
fn loom_handlers_in_any_order_pass_the_first_failure_down_the_chain() {
    loom::model(|| {
        let pool = Arc::new(FailurePool::default());
        let [first, second, third] = finish_concurrently(chain(&pool, [true, false, false]));
        let Some(Failed::Kernel(root)) = first else {
            panic!("the first command buffer did not report its failure");
        };
        assert_eq!(dependency(second).as_ref(), Some(&root));
        assert_eq!(dependency(third).as_ref(), Some(&root));
        assert_eq!(lock(&pool.0).len(), 3);
    });
}

#[test]
fn loom_handlers_in_any_order_report_a_failure_inside_the_chain_downstream_only() {
    loom::model(|| {
        let pool = Arc::new(FailurePool::default());
        let [first, second, third] = finish_concurrently(chain(&pool, [false, true, false]));
        assert!(first.is_none());
        let Some(Failed::Kernel(root)) = second else {
            panic!("the second command buffer did not report its failure");
        };
        assert_eq!(dependency(third).as_ref(), Some(&root));
        assert_eq!(lock(&pool.0).len(), 3);
    });
}
