#![no_std]
#![feature(core_intrinsics)]
#![allow(internal_features)]

use core::intrinsics::{is_val_statically_known, likely, unlikely};
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn hints(out: WriteBuffer<u32>, value: u32, divisor: u32) {
    let known = is_val_statically_known(value);
    let quotient = if likely(divisor != 0) { value / divisor } else { 0 };
    let remainder = if unlikely(divisor == 0) { 0 } else { value % divisor };
    unsafe {
        out.store_unchecked(0, known as u32);
        out.store_unchecked(1, quotient);
        out.store_unchecked(2, remainder);
    }
}
