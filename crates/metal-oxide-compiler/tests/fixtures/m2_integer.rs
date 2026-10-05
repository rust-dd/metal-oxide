#![no_std]

use core::ops::Add;
use metal_oxide_device::{WriteBuffer, kernel};

fn add<T: Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

#[kernel]
pub unsafe fn integer_math(out: WriteBuffer<i32>, a: i32, b: i32, shift: u32) {
    // SAFETY: the caller supplies six writable elements and launches one thread.
    unsafe {
        out.store_unchecked(0, add(a, b));
        out.store_unchecked(1, a - b);
        out.store_unchecked(2, a * b);
        out.store_unchecked(3, -a);
        out.store_unchecked(4, a << shift);
        out.store_unchecked(5, a >> shift);
    }
}
