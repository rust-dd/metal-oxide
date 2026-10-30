extern crate std;

use super::Float;

#[test]
fn powi_repeats_compiler_rt_squaring_bit_for_bit() {
    let bases = [
        0.0_f32,
        -0.0,
        1.0,
        -1.0,
        1.5,
        -2.25,
        0.1,
        1.0 + f32::EPSILON,
        3.0e-20,
        1.0e10,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let exponents = [
        0,
        1,
        2,
        3,
        7,
        -1,
        -2,
        -3,
        31,
        -31,
        127,
        255,
        1 << 20,
        i32::MAX,
        i32::MIN,
        i32::MIN + 1,
    ];
    for base in bases {
        for n in exponents {
            // std's powi calls compiler-rt's __powisf2 when the exponent is not a constant.
            let expected = std::hint::black_box(base).powi(std::hint::black_box(n));
            let actual = Float::powi(base, n);
            assert!(
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
                "{base:e}^{n}: {actual:e}, std {expected:e}"
            );
        }
    }
    assert!(Float::powi(f32::NAN, 0) == 1.0 && Float::powi(f32::NAN, 2).is_nan());
}
