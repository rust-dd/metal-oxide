#![no_std]

use metal_oxide_device::{WriteBuffer, kernel, thread_idx, threadgroup};

fn enabled(mode: u32, unused: u32) -> bool {
    let _ = unused;
    mode != 0
}

unsafe fn conditional_barrier(condition: bool, unused: u32) {
    let _ = unused;
    if condition {
        unsafe { threadgroup::barrier() }
    }
}

unsafe fn synchronize(mode: u32, lane: u32) {
    unsafe { conditional_barrier(enabled(mode, lane), lane) }
}

#[cfg(not(varying_return))]
#[kernel(block = (32, 1, 1))]
pub unsafe fn uniform_helpers(output: WriteBuffer<u32>, mode: u32) {
    let lane = thread_idx().x;
    let shared = threadgroup::shared::<u32, 32>();
    unsafe {
        shared.store_unchecked(lane, lane);
        #[cfg(not(varying))]
        synchronize(mode, lane);
        #[cfg(varying)]
        synchronize(lane, lane);
        let index = if mode != 0 { (lane + 1) & 31 } else { lane };
        output.store_unchecked(lane, shared.load_unchecked(index));
    }
}

#[cfg(varying_return)]
unsafe fn early_return(condition: bool) {
    if condition {
        return;
    }
    unsafe { threadgroup::barrier() }
}

#[cfg(varying_return)]
#[kernel]
pub unsafe fn divergent_helper() {
    unsafe { early_return(thread_idx().x == 0) }
}
