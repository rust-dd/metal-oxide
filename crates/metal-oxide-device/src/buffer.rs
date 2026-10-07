/// A read-only device buffer handle supplied by the kernel launch bindings.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ReadBuffer<T> {
    ptr: *const T,
}

impl<T: Copy> ReadBuffer<T> {
    /// Reads an element without checking its index.
    ///
    /// # Safety
    ///
    /// The index must address an initialized element in this buffer. The read
    /// must not race with a write to that element.
    pub unsafe fn load_unchecked(self, index: u32) -> T {
        let mut out = core::mem::MaybeUninit::<T>::uninit();
        // SAFETY: the caller guarantees bounds, initialization, and access discipline.
        unsafe {
            crate::intrinsics::__metal_buffer_load(
                self.ptr.cast(),
                index,
                out.as_mut_ptr().cast(),
                core::mem::size_of::<T>(),
            );
            out.assume_init()
        }
    }
}

/// A writable device buffer handle supplied by the kernel launch bindings.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct WriteBuffer<T> {
    ptr: *mut T,
}

impl<T: Copy> WriteBuffer<T> {
    /// Writes an element without checking its index.
    ///
    /// # Safety
    ///
    /// The index must address an element in this buffer. The write must not race
    /// with any other access to that element.
    pub unsafe fn store_unchecked(self, index: u32, value: T) {
        // SAFETY: the caller guarantees bounds and exclusive access to the element.
        unsafe {
            crate::intrinsics::__metal_buffer_store(
                self.ptr.cast(),
                index,
                (&raw const value).cast(),
                core::mem::size_of::<T>(),
            );
        }
    }
}
