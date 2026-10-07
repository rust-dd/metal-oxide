#![no_std]

use metal_oxide_device::{WriteBuffer, kernel, simdgroup, thread_idx};

unsafe fn sum(value: f32) -> f32 {
    // SAFETY: this helper requires uniform participation from its caller.
    unsafe { simdgroup::sum(value) }
}

#[kernel]
pub unsafe fn divergent_return(output: WriteBuffer<f32>) {
    let lane = thread_idx().x;
    let value = match lane {
        0 => return,
        1 => 2.0,
        _ => 3.0,
    };
    // SAFETY: output covers the block; the compiler must reject the skipped SIMD call.
    unsafe {
        output.store_unchecked(lane, sum(value));
    }
}
