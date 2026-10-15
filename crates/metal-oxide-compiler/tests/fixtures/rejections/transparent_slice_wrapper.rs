#![no_std]
use core::{marker::PhantomData, ptr::NonNull};
use metal_oxide_device::kernel;

#[repr(transparent)]
struct View(NonNull<[u32]>, PhantomData<u8>);

#[kernel]
pub unsafe fn invalid() {
    let mut data = [1u32, 2];
    let _view = View(NonNull::from(&mut data[..]), PhantomData);
}
