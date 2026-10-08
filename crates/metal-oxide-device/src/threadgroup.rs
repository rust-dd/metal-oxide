/// A handle to one statically allocated threadgroup buffer.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ThreadgroupBuffer<T, const N: usize> {
    ptr: *mut T,
}

/// Creates uninitialized storage shared by threads in the current block.
/// Each call site has distinct storage; allocation inside loops is unsupported.
pub fn shared<T: Copy, const N: usize>() -> ThreadgroupBuffer<T, N> {
    // SAFETY: the compiler allocates a distinct, aligned buffer at this call site.
    let ptr = unsafe {
        crate::intrinsics::__metal_threadgroup_alloc(
            core::mem::size_of::<T>() * N,
            core::mem::align_of::<T>(),
        )
    };
    ThreadgroupBuffer { ptr: ptr.cast() }
}

impl<T: Copy, const N: usize> ThreadgroupBuffer<T, N> {
    /// # Safety
    /// The index must be below N, initialized, and synchronized with other accesses.
    pub unsafe fn load_unchecked(self, index: u32) -> T {
        let mut out = core::mem::MaybeUninit::<T>::uninit();
        // SAFETY: the caller guarantees bounds, initialization, and synchronization.
        unsafe {
            crate::intrinsics::__metal_threadgroup_load(
                self.ptr.cast(),
                index,
                out.as_mut_ptr().cast(),
                core::mem::size_of::<T>(),
            );
            out.assume_init()
        }
    }

    /// # Safety
    /// The index must be below N and this write must not race with other accesses.
    pub unsafe fn store_unchecked(self, index: u32, value: T) {
        // SAFETY: the caller guarantees bounds and synchronization.
        unsafe {
            crate::intrinsics::__metal_threadgroup_store(
                self.ptr.cast(),
                index,
                (&raw const value).cast(),
                core::mem::size_of::<T>(),
            );
        }
    }
}

/// Orders threadgroup-memory accesses and waits for every thread in the block.
///
/// # Safety
/// All threads in the block must reach the same barrier in the same loop iteration.
pub unsafe fn barrier() {
    // SAFETY: the caller guarantees uniform block participation.
    unsafe { crate::intrinsics::__metal_threadgroup_barrier() }
}

/// Allocates uninitialized `u32` or `i32` atomic storage for one threadgroup.
/// Initialize each cell with an atomic store, then synchronize before reading.
/// Each call site has distinct storage; allocation inside loops is unsupported.
pub fn shared_atomic<T: Copy, const N: usize>() -> crate::AtomicThreadgroupBuffer<T, N> {
    // SAFETY: the compiler allocates distinct, aligned atomic storage at this call site.
    let ptr = unsafe {
        crate::intrinsics::__metal_threadgroup_alloc(
            core::mem::size_of::<T>() * N,
            core::mem::align_of::<T>(),
        )
    };
    crate::AtomicThreadgroupBuffer { ptr: ptr.cast() }
}
