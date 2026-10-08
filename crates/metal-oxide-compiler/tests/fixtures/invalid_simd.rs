#![no_std]
use metal_oxide_device::{WriteBuffer, kernel, simdgroup, thread_idx};
unsafe fn scan(value: u32, enabled: bool) -> u32 {
    if enabled { unsafe { simdgroup::exclusive_sum(value) } } else { 0 }
}
unsafe fn relative(value: u32, delta: u32) -> u32 {
    unsafe { simdgroup::shuffle_up(value, delta) }
}
#[kernel]
pub unsafe fn invalid(out: WriteBuffer<u32>) {
    let lane = thread_idx().x;
    #[cfg(divergent)] let value = unsafe { scan(lane, lane % 2 == 0) };
    #[cfg(control)] let value = unsafe { relative(lane, lane) };
    #[cfg(lane)] let value = unsafe { simdgroup::shuffle(lane, 64) };
    #[cfg(float_bits)] let value = unsafe { simdgroup::xor(1.0_f32) } as u32;
    #[cfg(narrow)] let value = unsafe { simdgroup::sum(1_u16) } as u32;
    unsafe { out.store_unchecked(lane, value); }
}
