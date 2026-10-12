#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel, thread_idx};

/// # Safety
/// `bounds` holds a start/end pair and `sums`/`counts` one value per thread.
#[kernel]
pub unsafe fn range_sums(bounds: ReadBuffer<i32>, sums: WriteBuffer<i32>, counts: WriteBuffer<u32>) {
    let i = thread_idx().x;
    let (start, end) = unsafe { (bounds.load_unchecked(2 * i), bounds.load_unchecked(2 * i + 1)) };
    let mut sum = 0;
    let mut count = 0;
    for k in start..end {
        sum += k;
        count += 1;
    }
    unsafe {
        sums.store_unchecked(i, sum);
        counts.store_unchecked(i, count);
    }
}

/// # Safety
/// `output` holds 16 values per thread.
#[kernel]
pub unsafe fn nested_ranges(output: WriteBuffer<u32>, offset: u32) {
    let i = thread_idx().x;
    let mut table = [0_u32; 16];
    for row in 0..4 {
        for column in row..4 {
            table[row * 4 + column] = (row * 10 + column) as u32 + offset;
        }
    }
    for k in 0..16 {
        unsafe { output.store_unchecked(16 * i + k as u32, table[k]) };
    }
}

/// # Safety
/// `limits` holds one value and `output` three values per thread.
#[kernel]
pub unsafe fn byte_ranges(limits: ReadBuffer<u8>, output: WriteBuffer<u32>) {
    let i = thread_idx().x;
    let limit = unsafe { limits.load_unchecked(i) };
    let mut sums = [0_u32; 3];
    for k in 250_u8..255 {
        sums[0] += k as u32;
    }
    for k in limit..u8::MAX {
        sums[1] += k as u32 + 1;
    }
    for k in 5_u8..limit {
        sums[2] += k as u32 * 3;
    }
    for k in 0..3 {
        unsafe { output.store_unchecked(3 * i + k as u32, sums[k]) };
    }
}

/// # Safety
/// `output` holds one value per thread.
#[kernel]
pub unsafe fn manual_next(output: WriteBuffer<u32>, n: u32) {
    let i = thread_idx().x;
    let mut range = i..n;
    let mut product = 1_u32;
    while let Some(k) = range.next() {
        product *= k + 2;
    }
    unsafe { output.store_unchecked(i, product) };
}
