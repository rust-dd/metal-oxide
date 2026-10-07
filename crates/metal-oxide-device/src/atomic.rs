/// A device buffer accessed through relaxed integer atomic operations.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AtomicBuffer<T> {
    ptr: *mut T,
}

impl<T: Copy> AtomicBuffer<T> {
    /// Adds a value atomically and returns the previous value. Supports i32 and u32.
    ///
    /// # Safety
    /// The index must be in bounds and initialized. Concurrent accesses must all
    /// be atomic; this operation does not order accesses to other memory.
    pub unsafe fn fetch_add_relaxed(self, index: u32, value: T) -> T {
        let mut out = core::mem::MaybeUninit::<T>::uninit();
        // SAFETY: the caller guarantees bounds, initialization, and atomic access.
        unsafe {
            crate::intrinsics::__metal_atomic_add(
                self.ptr.cast(),
                index,
                (&raw const value).cast(),
                out.as_mut_ptr().cast(),
                core::mem::size_of::<T>(),
            );
            out.assume_init()
        }
    }
}
