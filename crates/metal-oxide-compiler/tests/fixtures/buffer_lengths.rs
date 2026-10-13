#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

fn total(values: &[f32]) -> f32 {
    let mut sum = 0.0;
    for i in 0..values.len() {
        sum += values[i];
    }
    sum
}

/// Writes `a[i] + b[i]` where all three buffers have an element `i`, and the sum of `a` to
/// the last element of `out`.
///
/// # Safety
/// Each thread writes only its own element of `out`, and one thread the last.
#[kernel]
pub unsafe fn lengths(a: ReadBuffer<f32>, b: ReadBuffer<f32>, out: WriteBuffer<f32>) {
    let i = (block_idx().x * block_dim().x + thread_idx().x) as usize;
    let left = a.as_slice();
    let right = b.as_slice();
    if i < left.len() && i < right.len() && i < out.len() {
        unsafe { out.store_unchecked(i as u32, left[i] + right[i]) };
    }
    if i == 0 && !out.is_empty() && !a.is_empty() {
        unsafe { out.store_unchecked((out.len() - 1) as u32, total(left)) };
    }
}
