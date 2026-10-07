#![no_std]

use metal_oxide_device::{WriteBuffer, kernel, thread_idx, threadgroup};

unsafe fn synchronize() {
    // SAFETY: this helper requires uniform participation from its caller.
    unsafe {
        threadgroup::barrier();
    }
}

#[kernel]
pub unsafe fn divergent_match(output: WriteBuffer<u32>) {
    let lane = thread_idx().x;
    // SAFETY: output covers the block; the compiler must reject the nonuniform barrier.
    unsafe {
        match lane {
            0 => synchronize(),
            1 => output.store_unchecked(lane, 11),
            _ => output.store_unchecked(lane, 22),
        }
    }
}
