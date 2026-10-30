use super::transcendentals::{flush, matches, same_bits_or_nan};
use super::*;
use metal_oxide::{BF16, F16, GpuValue};
use metal_oxide_artifact::MathMode;

const FLOAT: &str = "crates/metal-oxide-compiler/tests/fixtures/float_methods.rs";
const NAMES: [&str; 10] = [
    "exp2", "log2", "log10", "tan", "asin", "acos", "atan2", "rsqrt", "hypot", "powi",
];

/// Zero signs, infinities, NaN, domain edges, strided bit patterns and an even sweep.
fn cases() -> Vec<(f32, f32, i32)> {
    let special = [
        0.0_f32,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        core::f32::consts::FRAC_PI_2,
        -core::f32::consts::PI,
        1.0e5,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut cases = special
        .iter()
        .flat_map(|&x| special.iter().map(move |&y| (x, y)))
        .enumerate()
        .map(|(i, (x, y))| (x, y, i as i32 % 9 - 4))
        .collect::<Vec<_>>();
    cases.extend((0..=128).map(|i| (i as f32 / 64.0 - 1.0, 0.75, i - 64)));
    cases.extend((0..1024).map(|i| {
        (
            (i as f32 - 512.0) * 0.37,
            1.25 - (i % 7) as f32,
            i % 41 - 20,
        )
    }));
    cases.extend((0..1 << 14).map(|k: u32| {
        let x = f32::from_bits(k.wrapping_mul(262_147) ^ 0x5a5a);
        let y = f32::from_bits(k.wrapping_mul(65_537) ^ 0x3c3c_0000);
        (x, y, (k % 67) as i32 - 33)
    }));
    cases
}

/// The f64 results of the methods with a Metal bound, rounded to f32, and that bound in ULPs.
fn expected(x: f32, y: f32) -> [(f32, u32); 9] {
    let (x, y) = (x as f64, y as f64);
    [
        (x.exp2(), 4),
        (x.log2(), 4),
        (x.log10(), 4),
        (x.tan(), 6),
        (x.asin(), 4),
        (x.acos(), 4),
        (x.atan2(y), 6),
        (1.0 / x.sqrt(), 2),
        (x.hypot(y), 2),
    ]
    .map(|(value, ulps)| (value as f32, ulps))
}

/// compiler-rt's squaring loop with each subnormal operand and result flushed to zero, as
/// Apple GPUs compute it.
fn flushed_powi(x: f32, n: i32) -> f32 {
    let mut base = flush(x);
    let mut exponent = n.unsigned_abs();
    let mut result = 1.0_f32;
    loop {
        if exponent & 1 != 0 {
            result = flush(result * base);
        }
        exponent >>= 1;
        if exponent == 0 {
            break;
        }
        base = flush(base * base);
    }
    if n < 0 { flush(1.0 / result) } else { result }
}

fn run(
    device: &Device,
    mode: MathMode,
    cases: &[(f32, f32, i32)],
) -> metal_oxide::Result<Vec<[f32; 10]>> {
    let pipeline = pipeline_with_math_mode(device, FLOAT, "float_methods", mode)?;
    let input = device.buffer_from_slice(cases)?;
    let mut output = device.buffer_zeroed::<[f32; 10]>(cases.len())?;
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
    Ok(output.read().wait().to_vec())
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn float_methods_match_rust_within_metal_precise_bounds() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let cases = cases();
    let results = run(&device, MathMode::Precise, &cases)?;
    let mut failures = Vec::new();
    for (&(x, y, k), row) in cases.iter().zip(&results) {
        let exact = expected(x, y);
        let flushed = expected(flush(x), flush(y));
        for (i, (&(want, ulps), &(flushed, _))) in exact.iter().zip(&flushed).enumerate() {
            if !matches(row[i], want, ulps) && !matches(row[i], flushed, ulps) {
                failures.push(format!(
                    "{}({x:e}, {y:e}) = {:e}, expected {want:e}",
                    NAMES[i], row[i]
                ));
            }
        }
        // std's powi runs compiler-rt's squaring loop when the exponent is not a constant.
        let powi = [
            std::hint::black_box(x).powi(std::hint::black_box(k)),
            flushed_powi(x, k),
        ];
        if !powi.iter().any(|&want| same_bits_or_nan(row[9], want)) {
            failures.push(format!(
                "powi({x:e}, {k}) = {:e}, expected {:e}",
                row[9], powi[0]
            ));
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
fn fast_float_methods_stay_close_to_rust_on_moderate_inputs() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let cases = (0..=256)
        .map(|i| (i as f32 / 128.0 - 1.0, 0.25 + i as f32 / 64.0, i % 9 - 4))
        .collect::<Vec<_>>();
    let results = run(&device, MathMode::Fast, &cases)?;
    for (&(x, y, k), row) in cases.iter().zip(&results) {
        let mut want = expected(x, y).map(|(value, _)| value).to_vec();
        want.push(x.powi(k));
        for (i, (&actual, &want)) in row.iter().zip(&want).enumerate() {
            let close = (actual - want).abs() <= 1e-4 * want.abs().max(1.0)
                || (actual.is_nan() && want.is_nan())
                || actual == want;
            assert!(
                close,
                "fast {}({x}, {y}, {k}) = {actual}, expected {want}",
                NAMES[i]
            );
        }
    }
    Ok(())
}

/// A 16-bit float that the host converts exactly to and from f64.
trait Half: GpuValue + Copy + std::fmt::Debug {
    /// The bits of the smallest positive normal value.
    const MIN_NORMAL: u16;
    fn from_bits(bits: u16) -> Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn bits(self) -> u16;
}

macro_rules! half {
    ($ty:ty, $min_normal:expr) => {
        impl Half for $ty {
            const MIN_NORMAL: u16 = $min_normal;
            fn from_bits(bits: u16) -> Self {
                Self::from_bits(bits)
            }
            fn from_f64(value: f64) -> Self {
                Self::from_f64(value)
            }
            fn to_f64(self) -> f64 {
                self.to_f64()
            }
            fn bits(self) -> u16 {
                self.to_bits()
            }
        }
    };
}
half!(F16, 0x0400);
half!(BF16, 0x0080);

/// Whether `actual` is `want` rounded or its neighbor: an f32 result within Metal's bounds can
/// round to either side of a 16-bit midpoint, and subnormal results may flush to zero.
fn rounded<T: Half>(actual: T, want: f64) -> bool {
    let want = T::from_f64(want);
    let (a, b) = (actual.to_f64(), want.to_f64());
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    let ordered = |bits: u16| {
        if bits & 0x8000 == 0 {
            bits | 0x8000
        } else {
            !bits
        }
    };
    ordered(actual.bits()).abs_diff(ordered(want.bits())) <= 1
        || (a == 0.0 && b.abs() < T::from_bits(T::MIN_NORMAL).to_f64())
}

/// Whether `actual` is the operand that `f64::min` or `f64::max` chose; zeros of either sign
/// compare equal.
fn chosen<T: Half>(actual: T, want: f64) -> bool {
    let a = actual.to_f64();
    (a.is_nan() && want.is_nan()) || a == want
}

fn sixteen_bit_methods<T: Half>(
    device: &Device,
    module: &Module,
    name: &str,
) -> metal_oxide::Result<()> {
    let pipeline = Pipeline::new(device, module, name)?;
    let values = (0..4096_u32)
        .map(|i| {
            [
                T::from_bits(i.wrapping_mul(0x9e37) as u16),
                T::from_bits(i.wrapping_mul(0x79b9).wrapping_add(17) as u16),
            ]
        })
        .chain(
            [
                [0.0, -0.0],
                [-0.0, 1.0],
                [1.0, f64::NAN],
                [f64::NAN, 2.0],
                [f64::INFINITY, f64::NAN],
            ]
            .map(|pair| pair.map(T::from_f64)),
        )
        .collect::<Vec<_>>();
    let input = device.buffer_from_slice(&values)?;
    let mut output = device.buffer_zeroed::<[T; 12]>(values.len())?;
    // SAFETY: the safe kernel bounds its writes and checks the input length.
    unsafe {
        Launch::new(
            Ok(&pipeline),
            LaunchConfig::<64>::for_elements(values.len() as u32)?,
            (&input, &mut output),
            |(input, output)| Ok([Argument::read(input), Argument::write(output)]),
        )
    }
    .wait()?;
    // The methods compute in f32, where Apple GPUs flush subnormal operands and results.
    let flush = |value: f64| {
        if value.abs() < f32::MIN_POSITIVE as f64 {
            0.0_f64.copysign(value)
        } else {
            value
        }
    };
    let methods = |a: f64, b: f64| {
        [
            a.sqrt(),
            a.exp(),
            a.ln(),
            a.sin(),
            a.cos(),
            a.atan2(b),
            a.hypot(b),
            1.0 / (a * a * a),
            1.0 / a.sqrt(),
        ]
    };
    let mut failures = Vec::new();
    for (pair, row) in values.iter().zip(output.read().wait().iter()) {
        let [a, b] = pair.map(T::to_f64);
        let mut flushed = methods(flush(a), flush(b));
        flushed[7] = flushed_powi(a as f32, -3) as f64;
        for (i, (&exact, &flushed)) in methods(a, b).iter().zip(&flushed).enumerate() {
            if !rounded(row[i], exact) && !rounded(row[i], flushed) {
                failures.push(format!(
                    "{name} method {i}({a}, {b}) = {:?}, expected {exact}",
                    row[i]
                ));
            }
        }
        if row[9].bits() != pair[0].bits() & 0x7fff {
            failures.push(format!("{name} abs({a}) = {:?}", row[9]));
        }
        // Operands that flush to equal values compare equal, so either one may be chosen.
        let either = |actual: T| flush(a) == flush(b) && (chosen(actual, a) || chosen(actual, b));
        let minimum = chosen(row[10], a.min(b)) || either(row[10]);
        let maximum = chosen(row[11], a.max(b)) || either(row[11]);
        if !minimum || !maximum {
            failures.push(format!(
                "{name} min/max({a}, {b}) = {:?}, {:?}",
                row[10], row[11]
            ));
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
fn half_methods_round_f32_results_once() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/half_methods.rs",
        &[],
    );
    support::checked(output);
    let module = support::artifact::module(&device, &directory)?;
    sixteen_bit_methods::<F16>(&device, &module, "half_methods")
}

#[test]
#[ignore = "requires a Metal device on Apple Silicon"]
fn bfloat_methods_round_f32_results_once() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/bfloat_methods.rs",
        &[],
    );
    support::checked(output);
    let Some(module) = support::artifact::supported_module(&device, &directory, &["bfloat16"])?
    else {
        return Ok(());
    };
    sixteen_bit_methods::<BF16>(&device, &module, "bfloat_methods")
}
