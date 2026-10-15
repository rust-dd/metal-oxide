#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};
#[kernel]
pub unsafe fn reassigned_length(a: ReadBuffer<u32>, mut b: ReadBuffer<u32>, out: WriteBuffer<u32>) {
    if a.is_empty() { b = a; }
    unsafe { out.store_unchecked(0, b.len() as u32); }
}
