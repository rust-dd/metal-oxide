#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel, thread_idx};

#[kernel]
pub unsafe fn narrow_math(out8: WriteBuffer<u8>, out16: WriteBuffer<u16>, a: u8, b: u16, shift: u32) {
    // SAFETY: one thread writes six initialized elements in each distinct output.
    unsafe {
        out8.store_unchecked(0, a + 255);
        out8.store_unchecked(1, a - 255);
        out8.store_unchecked(2, a * 255);
        out8.store_unchecked(3, a << shift);
        out8.store_unchecked(4, a >> shift);
        out8.store_unchecked(5, !a);
        out16.store_unchecked(0, b + 65535);
        out16.store_unchecked(1, b - 65535);
        out16.store_unchecked(2, b * 65535);
        out16.store_unchecked(3, b << shift);
        out16.store_unchecked(4, b >> shift);
        out16.store_unchecked(5, !b);
    }
}

#[kernel]
pub unsafe fn widen(input: ReadBuffer<u8>, output: WriteBuffer<u16>, scale: u16, n: u32) {
    let i = thread_idx().x;
    if i < n {
        // SAFETY: one block reads and writes guarded separate n-element buffers.
        unsafe { output.store_unchecked(i, (input.load_unchecked(i) as u16) * scale); }
    }
}
