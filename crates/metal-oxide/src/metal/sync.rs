//! Synchronization primitives that loom models in `--cfg loom` test builds.

#[cfg(all(test, loom))]
pub(super) use loom::{
    cell::UnsafeCell,
    sync::{Condvar, Mutex, MutexGuard, atomic::AtomicBool},
};
#[cfg(not(all(test, loom)))]
pub(super) use std::sync::{Condvar, Mutex, MutexGuard, atomic::AtomicBool};

/// `std::cell::UnsafeCell` with loom's closure-scoped access methods.
#[cfg(not(all(test, loom)))]
pub(super) struct UnsafeCell<T>(std::cell::UnsafeCell<T>);

#[cfg(not(all(test, loom)))]
impl<T> UnsafeCell<T> {
    pub(super) fn new(value: T) -> Self {
        Self(std::cell::UnsafeCell::new(value))
    }

    pub(super) fn with<R>(&self, f: impl FnOnce(*const T) -> R) -> R {
        f(self.0.get())
    }

    pub(super) fn with_mut<R>(&self, f: impl FnOnce(*mut T) -> R) -> R {
        f(self.0.get())
    }
}

/// Every protected state stays consistent when a holder panics, so poisoning is ignored.
pub(super) fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}
