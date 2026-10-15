#![no_std]
use metal_oxide_device::{ReadBuffer, kernel};

struct View<'a> {
    items: &'a [u32],
}

#[kernel]
pub unsafe fn invalid(input: ReadBuffer<u32>) {
    let local = [1u32, 2];
    let mut view = View { items: input.as_slice() };
    view.items = &local;
}
