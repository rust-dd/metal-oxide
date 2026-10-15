#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

struct Tail<T: ?Sized> {
    head: u32,
    rest: T,
}

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, value: u32) {
    let sized = Tail { head: value, rest: [value, value] };
    let unsized_tail: &Tail<[u32]> = &sized;
    unsafe { output.store_unchecked(0, unsized_tail.head + unsized_tail.rest[0]) };
}
