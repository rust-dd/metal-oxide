#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn rejected(output: WriteBuffer<u32>, index: u32) {
    let values: &[u32; 0] = &[];
    let result = core::ops::Index::index(values, index as usize);
    // SAFETY: the caller supplies one output element and launches one thread.
    unsafe { output.store_unchecked(0, *result) };
}
