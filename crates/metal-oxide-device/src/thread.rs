/// Three-dimensional thread, block, or grid coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Returns the thread's coordinates within its block.
pub fn thread_idx() -> Dim3 {
    crate::intrinsics::__metal_thread_idx()
}

/// Returns the block's coordinates within the grid.
pub fn block_idx() -> Dim3 {
    crate::intrinsics::__metal_block_idx()
}

/// Returns the number of threads along each block axis.
pub fn block_dim() -> Dim3 {
    crate::intrinsics::__metal_block_dim()
}

/// Returns the number of blocks along each grid axis.
pub fn grid_dim() -> Dim3 {
    crate::intrinsics::__metal_grid_dim()
}

/// Returns thread-private array storage that is left unwritten.
///
/// # Safety
/// Each element must be written before it is read.
pub unsafe fn uninit_array<T: Copy, const N: usize>() -> [T; N] {
    let mut out = core::mem::MaybeUninit::<[T; N]>::uninit();
    // SAFETY: the caller writes each element before reading it.
    unsafe {
        crate::intrinsics::__metal_uninit(out.as_mut_ptr().cast(), core::mem::size_of::<[T; N]>());
        out.assume_init()
    }
}

/// The calling thread's position among all threads of the dispatch, which no other thread
/// shares. Only [`thread_index`] creates one, and it cannot be copied or cloned:
///
/// ```compile_fail,E0423
/// let forged = metal_oxide_device::ThreadIndex(0);
/// ```
///
/// ```compile_fail,E0382
/// let index = metal_oxide_device::thread_index();
/// let (first, second) = (index, index);
/// ```
///
/// ```compile_fail,E0599
/// let index = metal_oxide_device::thread_index().clone();
/// ```
pub struct ThreadIndex(());

impl ThreadIndex {
    /// `block_idx().x * block_dim().x + thread_idx().x`. A kernel that reads it, here or
    /// through `DisjointBuffer::get_mut`, accepts only one-dimensional launches.
    pub fn get(&self) -> usize {
        crate::intrinsics::__metal_thread_index()
    }
}

/// The calling thread's index, as a witness that only this thread holds.
pub fn thread_index() -> ThreadIndex {
    ThreadIndex(())
}
