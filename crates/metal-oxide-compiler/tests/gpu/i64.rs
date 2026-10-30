use super::*;
use metal_oxide_artifact::MathMode;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/i64.rs";
const NAMES: [&str; 37] = [
    "overflowing_add",
    "overflowing_mul",
    "overflowing_sub",
    "add overflow",
    "mul overflow",
    "sub overflow",
    "wrapping_shl",
    "wrapping_shr",
    "wrapping_neg",
    "not",
    "checked_div",
    "checked_rem",
    "checked_div_euclid",
    "checked_rem_euclid",
    "as f32 bits",
    "as i32",
    "as u8",
    "from i32",
    "wrapping_abs",
    "signum",
    "min",
    "max",
    "saturating_add",
    "saturating_sub",
    "saturating_mul",
    "abs_diff",
    "leading_zeros",
    "trailing_zeros",
    "count_ones",
    "rotate_left",
    "swap_bytes",
    "reverse_bits",
    "cmp",
    "midpoint",
    "i32 midpoint",
    "wrapping_pow",
    "match",
];

fn expected((narrow, a, b, shift): (i32, i64, i64, u32)) -> [i64; 37] {
    let (sum, add_overflow) = a.overflowing_add(b);
    let (product, mul_overflow) = a.overflowing_mul(b);
    let (difference, sub_overflow) = a.overflowing_sub(b);
    [
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
    ]
}

/// Every pair of edge values, then splitmix64 values of every magnitude.
fn cases() -> Vec<(i32, i64, i64, u32)> {
    let edges = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -7,
        i64::MIN,
        i64::MIN + 1,
        i64::MAX,
        i64::MAX - 1,
        1 << 32,
        -(1 << 32),
        i32::MAX as i64,
        i32::MIN as i64,
        0x0123_4567_89ab_cdef,
        -0x0123_4567_89ab_cdef,
    ];
    let shifts = [0, 1, 31, 32, 33, 63, 64, 65, u32::MAX];
    let narrows = [0, 1, -1, i32::MIN, i32::MAX, 12_345, -54_321];
    let mut cases = edges
        .iter()
        .flat_map(|&a| edges.iter().map(move |&b| (a, b)))
        .enumerate()
        .map(|(i, (a, b))| (narrows[i % narrows.len()], a, b, shifts[i % shifts.len()]))
        .collect::<Vec<_>>();
    let mut state = 0_u64;
    let mut next = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = state;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    };
    cases.extend((0..4096).map(|_| {
        let a = next() as i64 >> (next() % 64);
        let b = next() as i64 >> (next() % 64);
        (next() as i32, a, b, (next() % 70) as u32)
    }));
    cases
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn i64_operations_match_rust_bit_for_bit() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let pipeline = pipeline(&device, FIXTURE, "signed_wide_ops")?;
    let cases = cases();
    let input = device.buffer_from_slice(&cases)?;
    let mut output = device.buffer_zeroed::<[i64; 37]>(cases.len())?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(cases.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    let mut failures = Vec::new();
    for (&case, row) in cases.iter().zip(output.read().wait().iter()) {
        for (i, (&actual, want)) in row.iter().zip(expected(case)).enumerate() {
            if actual != want {
                failures.push(format!("{} {case:?}: {actual}, expected {want}", NAMES[i]));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn float_to_i64_casts_saturate_like_rust_in_each_math_mode() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let mut input = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.5,
        -0.0,
        0.0,
        1.5,
    ];
    for limit in [i64::MIN as f32, -(i64::MIN as f32)] {
        let bits = limit.to_bits();
        input.extend((bits - 4..=bits + 4).map(f32::from_bits));
    }
    input.extend((0..4096_u32).map(|i| f32::from_bits(i.wrapping_mul(0x9e37_79b9))));
    for mode in [MathMode::Precise, MathMode::Fast] {
        let pipeline = pipeline_with_math_mode(&device, FIXTURE, "float_to_signed_wide", mode)?;
        let buffer = device.buffer_from_slice(&input)?;
        let mut output = device.buffer_zeroed::<i64>(input.len())?;
        // SAFETY: the safe kernel bounds its writes and checks the input length.
        unsafe {
            Launch::new(
                Ok(&pipeline),
                LaunchConfig::<64>::for_elements(input.len() as u32)?,
                (&buffer, &mut output),
                |(buffer, output)| Ok([Argument::read(buffer), Argument::write(output)]),
            )
        }
        .wait()?;
        for (&actual, &input) in output.read().wait().iter().zip(&input) {
            assert_eq!(
                actual,
                input as i64,
                "{mode:?}, input={:08x}",
                input.to_bits()
            );
        }
    }
    Ok(())
}
