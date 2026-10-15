#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};
#[kernel]
pub unsafe fn slice_index_through_reference(out: WriteBuffer<u32>, index: u32) {
    let values = [1_u32, 2, 3, 4];
    let mut view: &[u32] = &values;
    let handle = &mut view;
    // The bounds check and the index each read their own copy of `*handle`.
    unsafe { out.store_unchecked(0, handle[index as usize]); }
}
