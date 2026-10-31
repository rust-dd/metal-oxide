#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn summed(out: WriteBuffer<u32>, value: u32) {
    // `sum` folds with the iterator's own `fold`, which reads its `MaybeUninit` elements.
    let total = [value, 1, 2].into_iter().sum::<u32>();
    unsafe { out.store_unchecked(0, total) };
}
