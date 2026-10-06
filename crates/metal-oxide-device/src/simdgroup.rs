/// The current lane within its SIMD group.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_simd_lane")]
pub fn lane_id() -> u32 {
    panic!("SIMD groups are only available in Metal kernels")
}

/// The number of lanes in a SIMD group.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_simd_size")]
pub fn size() -> u32 {
    panic!("SIMD groups are only available in Metal kernels")
}

/// The current SIMD group within its block.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_simd_group")]
pub fn group_id() -> u32 {
    panic!("SIMD groups are only available in Metal kernels")
}

/// The number of SIMD groups in the current block.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_simd_count")]
pub fn groups_per_block() -> u32 {
    panic!("SIMD groups are only available in Metal kernels")
}

/// Sums active lanes using Metal's parallel reduction order.
///
/// # Safety
/// All active lanes must reach the same operation in the same iteration.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_simd_sum")]
pub unsafe fn sum(_value: f32) -> f32 {
    panic!("SIMD groups are only available in Metal kernels")
}

/// Reads the value supplied by another lane.
///
/// # Safety
/// All active lanes must participate, and the source lane must be active and below size().
#[cfg_attr(
    target_env = "metal",
    rustc_diagnostic_item = "metal_oxide_simd_shuffle"
)]
pub unsafe fn shuffle(_value: f32, _lane: u32) -> f32 {
    panic!("SIMD groups are only available in Metal kernels")
}
