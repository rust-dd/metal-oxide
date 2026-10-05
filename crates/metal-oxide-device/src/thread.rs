/// Coordinates of a thread in its dispatch grid.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Returns the global thread coordinates inside a Metal kernel.
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_position_in_grid"
)]
pub fn position_in_grid() -> Dim3 {
    panic!("position_in_grid is only available in Metal kernels")
}
