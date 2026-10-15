#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};
// Every call has distinct argument facts, so no proof summary is reused.
fn inner(seed: u32) -> u32 {
    let (mut total, mut step) = (seed, 0);
    while step < 100 { total ^= step; step += 1; }
    total
}
fn middle(seed: u32) -> u32 {
    let (mut total, mut step) = (0, 0);
    while step < 64 { total ^= inner(seed * 64 + step); step += 1; }
    total
}
#[kernel]
pub unsafe fn proof_budget(out: WriteBuffer<u32>) {
    let (mut total, mut step) = (0, 0);
    while step < 64 { total ^= middle(step); step += 1; }
    unsafe { out.store_unchecked(0, total); }
}
