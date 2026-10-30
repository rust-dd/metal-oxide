#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn signed_wide_ops(
    input: ReadBuffer<(i32, i64, i64, u32)>,
    mut output: DisjointBuffer<[i64; 37]>,
) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let (narrow, a, b, shift) = input.as_slice()[index.get()];
        let (sum, add_overflow) = a.overflowing_add(b);
        let (product, mul_overflow) = a.overflowing_mul(b);
        let (difference, sub_overflow) = a.overflowing_sub(b);
        *output = [
            sum,
            product,
            difference,
            add_overflow as i64,
            mul_overflow as i64,
            sub_overflow as i64,
            a.wrapping_shl(shift),
            a.wrapping_shr(shift),
            a.wrapping_neg(),
            !a,
            a.checked_div(b).unwrap_or(0),
            a.checked_rem(b).unwrap_or(0),
            a.checked_div_euclid(b).unwrap_or(0),
            a.checked_rem_euclid(b).unwrap_or(0),
            (a as f32).to_bits() as i64,
            a as i32 as i64,
            a as u8 as i64,
            narrow as i64,
            a.wrapping_abs(),
            a.signum(),
            a.min(b),
            a.max(b),
            a.saturating_add(b),
            a.saturating_sub(b),
            a.saturating_mul(b),
            a.abs_diff(b) as i64,
            a.leading_zeros() as i64,
            a.trailing_zeros() as i64,
            a.count_ones() as i64,
            a.rotate_left(shift),
            a.swap_bytes(),
            a.reverse_bits(),
            a.cmp(&b) as i64,
            a.midpoint(b),
            narrow.midpoint(b as i32) as i64,
            a.wrapping_pow(shift),
            match a {
                i64::MIN => 7,
                -1 => 8,
                0x1_0000_0000 => 9,
                _ => 10,
            },
        ];
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn float_to_signed_wide(input: ReadBuffer<f32>, mut output: DisjointBuffer<i64>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = input.as_slice()[index.get()] as i64;
    }
}
