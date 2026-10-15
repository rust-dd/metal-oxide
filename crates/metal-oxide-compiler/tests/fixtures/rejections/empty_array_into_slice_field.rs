#![no_std]
use metal_oxide_device::kernel;

struct View<'a> {
    items: &'a [u32],
}

#[kernel]
pub unsafe fn invalid() {
    let data = [1u32, 2];
    let mut view = View { items: &data };
    let empty: &[u32; 0] = &[];
    view.items = empty;
}
