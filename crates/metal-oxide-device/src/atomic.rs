use crate::intrinsics;
use core::{
    mem::{MaybeUninit, size_of},
    sync::atomic::Ordering,
};

/// Device-memory atomics. The compiler checks supported types, operations and constant orderings.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AtomicBuffer<T> {
    ptr: *mut T,
}

/// Uninitialized atomic storage shared by one threadgroup.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AtomicThreadgroupBuffer<T, const N: usize> {
    pub(crate) ptr: *mut T,
}

macro_rules! modify {
    ($name:ident, $bridge:ident, $operation:literal) => {
        #[doc = concat!("Atomically ", $operation, ", returning the previous value. Integer arithmetic wraps.")]
        ///
        /// # Safety
        /// The index must be in bounds and initialized. Concurrent accesses must be atomic.
        pub unsafe fn $name(self, index: u32, value: T, order: Ordering) -> T {
            let mut out = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies valid atomic storage; the bridge initializes the output.
            unsafe {
                intrinsics::$bridge(self.ptr.cast(), index, (&raw const value).cast(),
                    out.as_mut_ptr().cast(), size_of::<T>(), order);
                out.assume_init()
            }
        }
    };
}

macro_rules! methods {
    () => {
        /// Loads an atomic element with a constant memory ordering.
        ///
        /// # Safety
        /// The index must be in bounds and initialized. Concurrent accesses must be atomic.
        pub unsafe fn load(self, index: u32, order: Ordering) -> T {
            let mut out = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies valid atomic storage; the bridge initializes the output.
            unsafe {
                intrinsics::__metal_atomic_load(
                    self.ptr.cast(),
                    index,
                    out.as_mut_ptr().cast(),
                    size_of::<T>(),
                    order,
                );
                out.assume_init()
            }
        }

        /// Stores an atomic element with a constant memory ordering.
        ///
        /// # Safety
        /// The index must be in bounds. Concurrent accesses must be atomic;
        /// initialize the element before an operation that reads its previous value.
        pub unsafe fn store(self, index: u32, value: T, order: Ordering) {
            // SAFETY: the caller supplies valid atomic storage.
            unsafe {
                intrinsics::__metal_atomic_store(
                    self.ptr.cast(),
                    index,
                    (&raw const value).cast(),
                    size_of::<T>(),
                    order,
                );
            }
        }

        /// Attempts one weak compare/exchange and returns `(observed, success)`.
        /// A failed attempt may be spurious. Floating values compare by their bit patterns.
        ///
        /// # Safety
        /// The index must be in bounds and initialized. Concurrent accesses must be atomic.
        pub unsafe fn compare_exchange_weak(
            self,
            index: u32,
            current: T,
            new: T,
            success: Ordering,
            failure: Ordering,
        ) -> (T, bool) {
            let mut observed = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies valid atomic storage; the bridge initializes the output.
            unsafe {
                let exchanged = intrinsics::__metal_atomic_compare_exchange_weak(
                    self.ptr.cast(),
                    index,
                    (&raw const current).cast(),
                    (&raw const new).cast(),
                    observed.as_mut_ptr().cast(),
                    size_of::<T>(),
                    (success, failure),
                );
                (observed.assume_init(), exchanged)
            }
        }

        modify!(exchange, __metal_atomic_exchange, "replaces the element");
        modify!(fetch_add, __metal_atomic_add, "adds to the element");
        modify!(fetch_sub, __metal_atomic_sub, "subtracts from the element");
        modify!(fetch_min, __metal_atomic_min, "takes the minimum");
        modify!(fetch_max, __metal_atomic_max, "takes the maximum");
        modify!(fetch_and, __metal_atomic_and, "applies bitwise AND");
        modify!(fetch_or, __metal_atomic_or, "applies bitwise OR");
        modify!(fetch_xor, __metal_atomic_xor, "applies bitwise XOR");
    };
}
impl<T: Copy> AtomicBuffer<T> {
    methods!();
}
impl<T: Copy, const N: usize> AtomicThreadgroupBuffer<T, N> {
    methods!();
}

impl AtomicBuffer<u64> {
    /// Atomically takes the minimum without returning the old value. Only relaxed ordering is supported.
    ///
    /// # Safety
    /// The index must be in bounds and initialized. Concurrent accesses must be atomic.
    pub unsafe fn min(self, index: u32, value: u64, order: Ordering) {
        // SAFETY: the caller supplies initialized atomic storage.
        unsafe { intrinsics::__metal_atomic_min_store(self.ptr.cast(), index, value, order) }
    }

    /// Atomically takes the maximum without returning the old value. Only relaxed ordering is supported.
    ///
    /// # Safety
    /// The index must be in bounds and initialized. Concurrent accesses must be atomic.
    pub unsafe fn max(self, index: u32, value: u64, order: Ordering) {
        // SAFETY: the caller supplies initialized atomic storage.
        unsafe { intrinsics::__metal_atomic_max_store(self.ptr.cast(), index, value, order) }
    }
}
