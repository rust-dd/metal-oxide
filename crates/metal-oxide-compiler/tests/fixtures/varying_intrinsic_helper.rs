#![no_std]

use metal_oxide_device::{Float, kernel, thread_idx, threadgroup};

fn root(value: f32) -> f32 {
    value.sqrt()
}

fn lane_value() -> f32 {
    if thread_idx().x == 0 { 1.0 } else { 4.0 }
}

#[kernel(block = (32, 1, 1))]
pub unsafe fn varying_intrinsic_helper() {
    if root(lane_value()) > 0.0 {
        unsafe { threadgroup::barrier() };
    }
}
