#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};
#[kernel]
pub unsafe fn divide_through_reference(out: WriteBuffer<i32>, mut value: i32, divisor: i32) {
    let quotient = &mut value;
    // The overflow check reads the dividend through the mutable reference.
    *quotient /= divisor;
    unsafe { out.store_unchecked(0, value); }
}
