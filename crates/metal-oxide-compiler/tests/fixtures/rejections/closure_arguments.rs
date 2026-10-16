#![no_std]
#![feature(fn_traits)]

use metal_oxide_device::{WriteBuffer, kernel};

const ARGUMENTS: (u32,) = (3,);

#[kernel]
pub unsafe fn rejected(output: WriteBuffer<u32>) {
    let closure = |value: u32| value + 1;
    let result = closure.call(ARGUMENTS);
    // SAFETY: the caller supplies one output element and launches one thread.
    unsafe { output.store_unchecked(0, result) };
}
