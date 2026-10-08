use crate::intrinsics;
use core::mem::{MaybeUninit, size_of};

/// Device-memory atomic handle. The compiler supports `u32` and `i32` elements.
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
        /// The index must be in bounds and initialized. Concurrent accesses must
        /// be atomic. Relaxed operations do not order accesses to other memory.
        pub unsafe fn $name(self, index: u32, value: T) -> T {
            let mut out = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies valid atomic storage; the bridge initializes each output.
            unsafe {
                intrinsics::$bridge(self.ptr.cast(), index, (&raw const value).cast(),
                    out.as_mut_ptr().cast(), size_of::<T>());
                out.assume_init()
            }
        }
    };
}

macro_rules! methods {
    () => {
        /// Loads an atomic element without ordering other memory accesses.
        ///
        /// # Safety
        /// The index must be in bounds and initialized. Concurrent accesses must be atomic.
        pub unsafe fn load_relaxed(self, index: u32) -> T {
            let mut out = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies valid atomic storage; the bridge initializes each output.
            unsafe {
                intrinsics::__metal_atomic_load(
                    self.ptr.cast(),
                    index,
                    out.as_mut_ptr().cast(),
                    size_of::<T>(),
                );
                out.assume_init()
            }
        }

        /// Stores an atomic element without ordering other memory accesses.
        ///
        /// # Safety
        /// The index must be in bounds. Concurrent accesses must be atomic;
        /// initialize the element before any operation that reads its previous value.
        pub unsafe fn store_relaxed(self, index: u32, value: T) {
            // SAFETY: the caller supplies valid atomic storage; the bridge initializes each output.
            unsafe {
                intrinsics::__metal_atomic_store(
                    self.ptr.cast(),
                    index,
                    (&raw const value).cast(),
                    size_of::<T>(),
                );
            }
        }

        /// Attempts one weak compare/exchange and returns `(observed, success)`.
        /// A failed attempt may be spurious; `observed` contains the value read.
        ///
        /// # Safety
        /// The index must be in bounds and initialized. Concurrent accesses must
        /// be atomic. Neither success nor failure orders accesses to other memory.
        pub unsafe fn compare_exchange_weak_relaxed(
            self,
            index: u32,
            current: T,
            new: T,
        ) -> (T, bool) {
            let mut observed = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies valid atomic storage; the bridge initializes each output.
            unsafe {
                let success = intrinsics::__metal_atomic_compare_exchange_weak(
                    self.ptr.cast(),
                    index,
                    (&raw const current).cast(),
                    (&raw const new).cast(),
                    observed.as_mut_ptr().cast(),
                    size_of::<T>(),
                );
                (observed.assume_init(), success)
            }
        }

        modify!(
            exchange_relaxed,
            __metal_atomic_exchange,
            "replaces the element"
        );
        modify!(fetch_add_relaxed, __metal_atomic_add, "adds to the element");
        modify!(
            fetch_sub_relaxed,
            __metal_atomic_sub,
            "subtracts from the element"
        );
        modify!(fetch_min_relaxed, __metal_atomic_min, "takes the minimum");
        modify!(fetch_max_relaxed, __metal_atomic_max, "takes the maximum");
        modify!(fetch_and_relaxed, __metal_atomic_and, "applies bitwise AND");
        modify!(fetch_or_relaxed, __metal_atomic_or, "applies bitwise OR");
        modify!(fetch_xor_relaxed, __metal_atomic_xor, "applies bitwise XOR");
    };
}
impl<T: Copy> AtomicBuffer<T> {
    methods!();
}
impl<T: Copy, const N: usize> AtomicThreadgroupBuffer<T, N> {
    methods!();
}
