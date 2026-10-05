/// A device buffer accessed through relaxed integer atomic operations.
#[repr(transparent)]
#[derive(Clone, Copy)]
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_atomic_buffer"
)]
pub struct AtomicBuffer<T> {
    _ptr: *mut T,
}

impl<T: Copy> AtomicBuffer<T> {
    /// Adds a value atomically and returns the previous value. Supports i32 and u32.
    ///
    /// # Safety
    /// The index must be in bounds and initialized. Concurrent accesses must all
    /// be atomic; this operation does not order accesses to other memory.
    #[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_atomic_add")]
    pub unsafe fn fetch_add_relaxed(self, _index: u32, _value: T) -> T {
        panic!("device atomics are only available in Metal kernels")
    }
}
