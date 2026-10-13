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
pub struct ThreadIndex(usize);

impl ThreadIndex {
    pub fn get(&self) -> usize {
        self.0
    }
}

/// The calling thread's index: `x + y * width + z * width * height` over the dispatch's threads.
pub fn thread_index() -> ThreadIndex {
    let (block, size, thread, grid) = (block_idx(), block_dim(), thread_idx(), grid_dim());
    let x = position(block.x, size.x, thread.x);
    let y = position(block.y, size.y, thread.y);
    let z = position(block.z, size.z, thread.z);
    let width = (grid.x as usize).wrapping_mul(size.x as usize);
    let height = (grid.y as usize).wrapping_mul(size.y as usize);
    // The runtime caps a dispatch at `u64::MAX` threads, so the index never wraps.
    ThreadIndex(x.wrapping_add(width.wrapping_mul(y.wrapping_add(height.wrapping_mul(z)))))
}

fn position(block: u32, size: u32, thread: u32) -> usize {
    (block as usize)
        .wrapping_mul(size as usize)
        .wrapping_add(thread as usize)
}
