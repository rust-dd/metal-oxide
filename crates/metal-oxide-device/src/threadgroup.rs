/// A handle to one statically allocated threadgroup buffer.
#[repr(transparent)]
#[derive(Clone, Copy)]
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_threadgroup_buffer"
)]
pub struct ThreadgroupBuffer<T, const N: usize> {
    ptr: *mut T,
}

/// Creates uninitialized storage shared by threads in the current block.
/// Each call site has distinct storage; allocation inside loops is unsupported.
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_threadgroup_alloc"
)]
pub fn shared<T: Copy, const N: usize>() -> ThreadgroupBuffer<T, N> {
    panic!("threadgroup storage is only available in Metal kernels")
}

impl<T: Copy, const N: usize> ThreadgroupBuffer<T, N> {
    /// # Safety
    /// The index must be below N, initialized, and synchronized with other accesses.
    #[cfg_attr(
        target_env = "metal",
        rustc_diagnostic_item = "metal_oxide_threadgroup_load"
    )]
    pub unsafe fn load_unchecked(self, index: u32) -> T {
        // SAFETY: the caller guarantees bounds, initialization, and synchronization.
        unsafe { self.ptr.add(index as usize).read() }
    }

    /// # Safety
    /// The index must be below N and this write must not race with other accesses.
    #[cfg_attr(
        target_env = "metal",
        rustc_diagnostic_item = "metal_oxide_threadgroup_store"
    )]
    pub unsafe fn store_unchecked(self, index: u32, value: T) {
        // SAFETY: the caller guarantees bounds and synchronization.
        unsafe { self.ptr.add(index as usize).write(value) }
    }
}

/// Orders threadgroup-memory accesses and waits for every thread in the block.
///
/// # Safety
/// All threads in the block must reach the same barrier in the same loop iteration.
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_threadgroup_barrier"
)]
pub unsafe fn barrier() {
    panic!("threadgroup barriers are only available in Metal kernels")
}
