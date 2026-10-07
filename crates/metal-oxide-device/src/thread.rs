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
