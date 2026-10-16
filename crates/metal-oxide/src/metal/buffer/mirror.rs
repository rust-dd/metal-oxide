use std::{cell::RefCell, ptr::NonNull};

use metal_oxide_artifact::Access;

use super::super::{
    access::AccessState,
    sync::{Mutex, UnsafeCell, lock},
};
use crate::GpuValue;

/// Which copy of a codec buffer holds its current values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::metal) enum Location {
    Cpu,
    Gpu,
    Synced,
}

/// The CPU values of a buffer whose elements use an explicit codec.
pub(in crate::metal) struct Mirror<T> {
    values: UnsafeCell<Vec<T>>,
    location: Mutex<Location>,
}

// SAFETY: values change only under the lock while stale. Staleness needs a GPU write, which
// needs `&mut Buffer`, so it is decoded at most once per shared borrow, before any slice exists.
unsafe impl<T: Send + Sync> Sync for Mirror<T> {}

impl<T: GpuValue> Mirror<T> {
    pub(in crate::metal) fn new(values: Vec<T>, location: Location) -> Self {
        Self {
            values: UnsafeCell::new(values),
            location: Mutex::new(location),
        }
    }

    /// Returns the CPU values, decoding the GPU copy first when a GPU write changed it.
    ///
    /// # Safety
    ///
    /// `bytes` must address this buffer's canonical values, with no recorded or pending GPU write.
    pub(in crate::metal) unsafe fn download(
        &self,
        access: &AccessState,
        bytes: NonNull<u8>,
    ) -> &[T] {
        let _entered = Entered::new(self);
        let mut location = lock(&self.location);
        if access.take_written() || *location == Location::Gpu {
            // A decoder panic leaves the GPU copy authoritative for the next read.
            *location = Location::Gpu;
            self.values.with_mut(|values| {
                // SAFETY: the lock serializes decoding; no slice exists while the copy is stale.
                let values = unsafe { &mut *values };
                // SAFETY: the caller provides initialized canonical bytes with no GPU writes.
                let bytes =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr(), values.len() * T::SIZE) };
                for (value, encoded) in values.iter_mut().zip(bytes.chunks_exact(T::SIZE)) {
                    *value = T::decode(encoded);
                }
            });
            *location = Location::Synced;
        }
        drop(location);
        // SAFETY: the values are current and stay unchanged while `self` is shared.
        self.values.with(|values| unsafe { &*values }).as_slice()
    }

    /// Returns the values for CPU mutation; the next dispatch encodes them.
    ///
    /// # Safety
    ///
    /// `bytes` must address this buffer's canonical values, with no recorded or pending GPU access.
    pub(in crate::metal) unsafe fn download_mut(
        &mut self,
        access: &AccessState,
        bytes: NonNull<u8>,
    ) -> &mut [T] {
        // SAFETY: forwarded from the caller.
        unsafe { self.download(access, bytes) };
        *lock(&self.location) = Location::Cpu;
        // SAFETY: `&mut self` excludes every other access to the values.
        self.values
            .with_mut(|values| unsafe { &mut *values })
            .as_mut_slice()
    }

    /// Encodes CPU-modified values into the GPU copy before a dispatch accesses it.
    ///
    /// # Safety
    ///
    /// `bytes` must address this buffer's canonical values. GPU work may access them only
    /// after this call; CPU-modified values guarantee that none is pending.
    pub(in crate::metal) unsafe fn prepare(&self, access: Access, bytes: NonNull<u8>) {
        let _entered = Entered::new(self);
        let mut location = lock(&self.location);
        if *location == Location::Cpu {
            self.values.with(|values| {
                // SAFETY: CPU mutation requires `&mut Buffer`; shared callers only read here.
                let values = unsafe { &*values };
                // SAFETY: the bytes are idle while the CPU copy is authoritative.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(bytes.as_ptr(), values.len() * T::SIZE)
                };
                for (value, encoded) in values.iter().zip(bytes.chunks_exact_mut(T::SIZE)) {
                    encoded.fill(0);
                    value.encode(encoded);
                }
            });
            *location = Location::Synced;
        }
        if access != Access::Read {
            *location = Location::Gpu;
        }
    }
}

#[cfg(not(all(test, loom)))]
std::thread_local! {
    static ENTERED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}
#[cfg(all(test, loom))]
loom::thread_local! {
    static ENTERED: RefCell<Vec<usize>> = RefCell::new(Vec::new());
}

/// Marks a mirror as being decoded or encoded on this thread, so a codec that
/// accesses its own buffer panics instead of deadlocking on the mirror lock.
struct Entered(usize);

impl Entered {
    fn new<T>(mirror: &Mirror<T>) -> Self {
        let id = std::ptr::from_ref(mirror).addr();
        ENTERED.with(|entered| {
            let mut entered = entered.borrow_mut();
            assert!(
                !entered.contains(&id),
                "a buffer codec cannot access its own buffer"
            );
            entered.push(id);
        });
        Self(id)
    }
}

impl Drop for Entered {
    fn drop(&mut self) {
        ENTERED.with(|entered| entered.borrow_mut().retain(|&id| id != self.0));
    }
}

#[cfg(all(test, not(loom)))]
mod tests;
