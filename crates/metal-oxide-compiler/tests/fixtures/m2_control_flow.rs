#![no_std]
use metal_oxide_device::{WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn control_flow(out: WriteBuffer<u32>, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i >= n { return; }
    let value = compute(i);
    unsafe { out.store_unchecked(i, value); }
}

fn compute(n: u32) -> u32 {
    let mut i = 0;
    let mut sum = 0;
    while i < n {
        if i & 1 == 0 { sum += 3; } else { sum += 7; }
        i += 1;
    }
    if n == 0 { return 99; }
    if sum > 10 {
        if sum > 30 { sum - 1 } else { sum + 2 }
    } else { sum }
}
