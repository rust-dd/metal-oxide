#![no_std]

use metal_oxide_device::{kernel, thread_idx, threadgroup};

#[kernel]
pub unsafe fn divergent_loop_exit() {
    let lane = thread_idx().x;
    let mut i = 0;
    while i < 4 {
        i += 1;
        match lane {
            0 if i == 2 => break,
            1 if i == 3 => continue,
            _ => {}
        }
        // SAFETY: intentionally invalid participation; this fixture must be rejected.
        unsafe {
            threadgroup::barrier();
        }
    }
}
