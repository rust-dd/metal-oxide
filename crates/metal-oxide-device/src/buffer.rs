use crate::ThreadIndex;

/// A read-only device buffer handle supplied by the kernel launch bindings.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ReadBuffer<T> {
    ptr: *const T,
}

impl<T: Copy> ReadBuffer<T> {
    /// The number of elements the host bound; kernel buffer parameters only.
    pub fn len(&self) -> usize {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast()) }
    }

    pub fn is_empty(&self) -> bool {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast()) == 0 }
    }

    /// The bound elements; kernel buffer parameters only.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the host bound `len` initialized elements that no thread writes.
        unsafe {
            core::slice::from_raw_parts(
                self.ptr,
                crate::intrinsics::__metal_buffer_length(self.ptr.cast()),
            )
        }
    }

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
    /// The number of elements the host bound; kernel buffer parameters only.
    pub fn len(&self) -> usize {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast_const().cast()) }
    }

    pub fn is_empty(&self) -> bool {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast_const().cast()) == 0 }
    }

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

/// A device buffer supporting both reads and writes within a kernel.
///
/// Launch bindings borrow its host buffer exclusively. Individual accesses must
/// still be in bounds and free of races between GPU threads.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct ReadWriteBuffer<T> {
    ptr: *mut T,
}

impl<T: Copy> ReadWriteBuffer<T> {
    /// The number of elements the host bound; kernel buffer parameters only.
    pub fn len(&self) -> usize {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast_const().cast()) }
    }

    pub fn is_empty(&self) -> bool {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast_const().cast()) == 0 }
    }

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

/// A writable buffer whose elements threads reach only through their own [`ThreadIndex`], so
/// writes of different threads never overlap.
///
/// ```compile_fail,E0382
/// fn copy(out: metal_oxide_device::DisjointBuffer<f32>) -> [metal_oxide_device::DisjointBuffer<f32>; 2] {
///     [out, out]
/// }
/// ```
///
/// ```compile_fail,E0499
/// use metal_oxide_device::{DisjointBuffer, ThreadIndex};
/// fn twice(out: &mut DisjointBuffer<f32>, index: &ThreadIndex) {
///     let first = out.get_mut(index);
///     let second = out.get_mut(index);
///     drop((first, second));
/// }
/// ```
#[repr(transparent)]
pub struct DisjointBuffer<T> {
    ptr: *mut T,
}

impl<T: Copy> DisjointBuffer<T> {
    /// The number of elements the host bound; kernel buffer parameters only.
    pub fn len(&self) -> usize {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast_const().cast()) }
    }

    pub fn is_empty(&self) -> bool {
        // SAFETY: the launch binds the length of every kernel buffer parameter.
        unsafe { crate::intrinsics::__metal_buffer_length(self.ptr.cast_const().cast()) == 0 }
    }

    /// The calling thread's element, or `None` past the end of the buffer.
    pub fn get_mut(&mut self, index: &ThreadIndex) -> Option<&mut T> {
        let index = index.get();
        // SAFETY: the index is below the bound length, and no other thread has it.
        unsafe {
            (index < crate::intrinsics::__metal_buffer_length(self.ptr.cast_const().cast()))
                .then(|| &mut *self.ptr.add(index))
        }
    }
}
