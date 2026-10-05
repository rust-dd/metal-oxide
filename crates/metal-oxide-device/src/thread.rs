/// Three-dimensional thread, block, or grid coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_dim3")]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Returns the thread's coordinates within its block.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_thread_idx")]
pub fn thread_idx() -> Dim3 {
    panic!("thread_idx is only available in Metal kernels")
}

/// Returns the block's coordinates within the grid.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_block_idx")]
pub fn block_idx() -> Dim3 {
    panic!("block_idx is only available in Metal kernels")
}

/// Returns the number of threads along each block axis.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_block_dim")]
pub fn block_dim() -> Dim3 {
    panic!("block_dim is only available in Metal kernels")
}

/// Returns the number of blocks along each grid axis.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_grid_dim")]
pub fn grid_dim() -> Dim3 {
    panic!("grid_dim is only available in Metal kernels")
}
