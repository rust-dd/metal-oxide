#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

macro_rules! bits {
    ($name:ident, $ty:ty) => {
        #[kernel]
        pub unsafe fn $name(
            input: ReadBuffer<$ty>,
            shifts: ReadBuffer<u32>,
            counts: WriteBuffer<u32>,
            values: WriteBuffer<$ty>,
            n: u32,
        ) {
            let i = block_idx().x * block_dim().x + thread_idx().x;
            if i < n {
                unsafe {
                    let a = input.load_unchecked(2 * i);
                    let b = input.load_unchecked(2 * i + 1);
                    let shift = shifts.load_unchecked(i);
                    counts.store_unchecked(3 * i, a.leading_zeros());
                    counts.store_unchecked(3 * i + 1, a.trailing_zeros());
                    counts.store_unchecked(3 * i + 2, a.count_ones());
                    values.store_unchecked(5 * i, a.reverse_bits());
                    values.store_unchecked(5 * i + 1, a.rotate_left(shift));
                    values.store_unchecked(5 * i + 2, a.rotate_right(shift));
                    values.store_unchecked(5 * i + 3, a.min(b));
                    values.store_unchecked(5 * i + 4, a.max(b));
                }
            }
        }
    };
}

bits!(bits_u8, u8);
bits!(bits_u16, u16);
bits!(bits_u32, u32);
bits!(bits_u64, u64);
bits!(bits_i8, i8);
bits!(bits_i16, i16);
bits!(bits_i32, i32);
