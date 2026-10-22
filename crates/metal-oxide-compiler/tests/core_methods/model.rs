pub const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/core_methods.rs";

pub const POWERS: &[(u32, i32, u32)] = &[
    (0, 0, 0),
    (0, 0, 1),
    (0, -1, u32::MAX),
    (1, 1, u32::MAX),
    (2, -2, 7),
    (2, -2, 31),
    (2, -2, 32),
    (3, -3, 19),
    (16, -16, 8),
    (255, i16::MIN as i32, 2),
    (65535, i32::MIN, 1),
    (u32::MAX, i32::MIN, 2),
    (u32::MAX, i32::MAX, 3),
];

pub fn powers(a: u32, b: i32, exponent: u32) -> [u32; 18] {
    [
        a.overflowing_pow(exponent).0,
        a.checked_pow(exponent).unwrap_or(0),
        a.checked_pow(exponent).is_some() as u32,
        a.wrapping_pow(exponent),
        a.saturating_pow(exponent),
        b.overflowing_pow(exponent).0 as u32,
        b.checked_pow(exponent).unwrap_or(0) as u32,
        b.checked_pow(exponent).is_some() as u32,
        b.wrapping_pow(exponent) as u32,
        b.saturating_pow(exponent) as u32,
        (a as u8).overflowing_pow(exponent).0 as u32,
        (a as u8).checked_pow(exponent).unwrap_or(0) as u32,
        (a as u8).wrapping_pow(exponent) as u32,
        (a as u8).saturating_pow(exponent) as u32,
        (b as i16).overflowing_pow(exponent).0 as u32,
        (b as i16).checked_pow(exponent).unwrap_or(0) as u32,
        (b as i16).wrapping_pow(exponent) as u32,
        (b as i16).saturating_pow(exponent) as u32,
    ]
}

pub const WIDE_POWERS: &[(u64, u32)] = &[
    (0, 0),
    (0, 1),
    (1, u32::MAX),
    (2, 63),
    (2, 64),
    (65537, 3),
    (1 << 32, 2),
    (u64::MAX, 2),
    (u64::MAX, 3),
];

pub fn wide_powers(a: u64, exponent: u32) -> [u64; 5] {
    [
        a.overflowing_pow(exponent).0,
        a.checked_pow(exponent).unwrap_or(0),
        a.checked_pow(exponent).is_some() as u64,
        a.wrapping_pow(exponent),
        a.saturating_pow(exponent),
    ]
}

pub const INTEGERS: &[(u32, u32, i32, i32)] = &[
    (0, 0, 0, 0),
    (0, 1, -1, 0),
    (1, 0, 0, -1),
    (7, 3, -7, 3),
    (3, 7, 7, -3),
    (u32::MAX, 2, i32::MIN, -1),
    (u32::MAX, 0, i32::MIN, 0),
    (u32::MAX, 1, i32::MIN, 1),
    (u32::MAX, u32::MAX, i32::MIN, i32::MAX),
    (1, 1, i32::MAX, i32::MIN),
];

pub fn comparisons(a: u32, b: u32, c: i32, d: i32) -> [i32; 7] {
    [
        a.cmp(&b) as i32,
        a.partial_cmp(&b).unwrap() as i32,
        c.cmp(&d) as i32,
        c.partial_cmp(&d).unwrap() as i32,
        (a, c).cmp(&(b, d)) as i32,
        (a, c).partial_cmp(&(b, d)).unwrap() as i32,
        c.signum(),
    ]
}

pub fn primitive_ordering(a: u64, b: u64) -> [i32; 10] {
    [
        (a as u8).cmp(&(b as u8)) as i32,
        (a as i8).cmp(&(b as i8)) as i32,
        (a as u16).cmp(&(b as u16)) as i32,
        (a as i16).cmp(&(b as i16)) as i32,
        (a as u32).cmp(&(b as u32)) as i32,
        (a as i32).cmp(&(b as i32)) as i32,
        a.cmp(&b) as i32,
        (a as usize).cmp(&(b as usize)) as i32,
        (a as isize).cmp(&(b as isize)) as i32,
        (a != 0).cmp(&(b != 0)) as i32,
    ]
}

pub fn checked_division(a: u32, b: u32, c: i32, d: i32) -> [u32; 8] {
    [
        a.checked_div(b).unwrap_or(0),
        a.checked_div(b).is_some() as u32,
        a.checked_rem(b).unwrap_or(0),
        a.checked_rem(b).is_some() as u32,
        c.checked_div(d).unwrap_or(0) as u32,
        c.checked_div(d).is_some() as u32,
        c.checked_rem(d).unwrap_or(0) as u32,
        c.checked_rem(d).is_some() as u32,
    ]
}

pub const FLOATS: &[u32] = &[
    0, 0x80000000, 0x3f800000, 0xbf800000, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
    0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7fc12345, 0x7f812345, 0xff812345,
];

pub const SUBNORMALS: &[u32] = &[1, 0x80000001, 0x007fffff, 0x807fffff];

pub fn float_methods(a: u32, b: u32) -> [u32; 4] {
    let (a, b) = (f32::from_bits(a), f32::from_bits(b));
    [
        a.copysign(b).to_bits(),
        a.signum().to_bits(),
        a.total_cmp(&b) as i32 as u32,
        a.partial_cmp(&b)
            .map(|order| order as i32 as u32)
            .unwrap_or(2),
    ]
}

pub const BITS: &[u64] = &[
    0,
    1,
    0x80,
    0x8000,
    0x80000000,
    0x0123456789abcdef,
    0xfedcba9876543210,
    1 << 32,
    1 << 63,
    u64::MAX,
];

pub const FLOAT_CLAMPS: &[(f32, f32, f32)] = &[
    (-3.0, -1.0, 1.0),
    (0.5, -1.0, 1.0),
    (3.0, -1.0, 1.0),
    (-0.0, 0.0, 0.0),
    (0.0, -0.0, -0.0),
    (f32::INFINITY, -1.0, 1.0),
    (f32::NEG_INFINITY, -1.0, 1.0),
    (f32::NAN, -1.0, 1.0),
    (1.0, 2.0, -2.0),
    (1.0, f32::NAN, 2.0),
    (1.0, -2.0, f32::NAN),
];

pub const UNSIGNED_CLAMPS: &[(u32, u32, u32)] = &[
    (0, 1, 7),
    (4, 1, 7),
    (u32::MAX, 1, 7),
    (u32::MAX, u32::MAX, u32::MAX),
    (4, 7, 1),
];

pub const SIGNED_CLAMPS: &[(i32, i32, i32)] = &[
    (i32::MIN, -1, 7),
    (4, -1, 7),
    (i32::MAX, -1, 7),
    (i32::MIN, i32::MIN, i32::MAX),
    (4, 7, -1),
];

pub fn bit_methods(value: u64) -> [u64; 11] {
    [
        (value as u8).swap_bytes() as u64,
        (value as i8).swap_bytes() as u64,
        (value as u16).swap_bytes() as u64,
        (value as i16).swap_bytes() as u64,
        (value as u32).swap_bytes() as u64,
        (value as i32).swap_bytes() as u64,
        value.swap_bytes(),
        (value as usize).swap_bytes() as u64,
        (value as isize).swap_bytes() as u64,
        (value as u32).leading_zeros() as u64,
        (value as u32).trailing_zeros() as u64,
    ]
}
