/// The current lane within its SIMD group.
pub fn lane_id() -> u32 {
    crate::intrinsics::__metal_simd_lane()
}

/// The number of lanes in a SIMD group.
pub fn size() -> u32 {
    crate::intrinsics::__metal_simd_size()
}

/// The current SIMD group within its block.
pub fn group_id() -> u32 {
    crate::intrinsics::__metal_simd_group()
}

/// The number of SIMD groups in the current block.
pub fn groups_per_block() -> u32 {
    crate::intrinsics::__metal_simd_count()
}

/// Sums active lanes using Metal's parallel reduction order.
///
/// # Safety
/// All active lanes must reach the same operation in the same iteration.
pub unsafe fn sum(value: f32) -> f32 {
    // SAFETY: the caller guarantees uniform participation.
    unsafe { crate::intrinsics::__metal_simd_sum(value) }
}

/// Reads the value supplied by another lane.
///
/// # Safety
/// All active lanes must participate, and the source lane must be active and below size().
pub unsafe fn shuffle(value: f32, lane: u32) -> f32 {
    // SAFETY: the caller guarantees participation and a valid source lane.
    unsafe { crate::intrinsics::__metal_simd_shuffle(value, lane) }
}
