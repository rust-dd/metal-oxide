#![no_std]
use metal_oxide_device::{WriteBuffer, kernel, simdgroup::{self, Matrix}};

#[kernel(block = (64, 1, 1))]
pub unsafe fn group_branch(output: WriteBuffer<f32>) {
    let group = simdgroup::group_id();
    let value = if group == 0 { Matrix::splat(1.0) } else { Matrix::splat(2.0) };
    // SAFETY: in a launch of one block, each SIMD group stores its own 8x8 block of 128 elements.
    unsafe { output.store_matrix(group * 8, 0, 8, value) };
}

#[kernel(block = (64, 1, 1))]
pub unsafe fn checked_group(output: WriteBuffer<f32>, groups: u32) {
    let group = simdgroup::group_id();
    assert!(group < groups, "SIMD group outside the launch contract");
    // SAFETY: in a launch of one block, each SIMD group stores its own 8x8 block of 128 elements.
    unsafe { output.store_matrix(group * 8, 0, 8, Matrix::splat(1.0)) };
}
