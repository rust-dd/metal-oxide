//! f32 device math. These functions have no native CPU implementation.
//!
//! Subnormal inputs and outputs may be flushed to signed zero. Arithmetic NaN
//! payloads and signaling behavior are unspecified. The documented edge cases
//! apply to the default precise mode, with Metal's precise accuracy bounds.
//!
//! `#[kernels(math = "fast")]` selects Metal's fast
//! math functions and permits reassociation, reciprocal approximations and
//! contraction. Zero signs and the precise edge cases below are not guaranteed
//! in that mode. Arithmetic still honors NaN and infinity, and float-to-integer
//! casts still saturate, mapping NaN to zero.
//!
//! Fast math has narrower accuracy domains: sin/cos have an absolute error bound
//! of 2^-13 within [-pi, pi]; ln requires a positive input; powf uses an exp2/log2
//! approximation and has no precise-mode guarantee for negative bases or 0^0.
//! See MSL sections 1.6.3 and 8.4 for the fast function contracts.

/// Square root using the selected numerical mode.
/// Subnormal inputs and outputs may be flushed to zero by the GPU.
pub fn sqrt(value: f32) -> f32 {
    crate::intrinsics::__metal_sqrt_f32(value)
}

/// Computes `a * b + c` with one rounding, without rounding the product first.
/// Subnormal inputs and outputs may be flushed to zero by the GPU.
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    crate::intrinsics::__metal_fma_f32(a, b, c)
}

/// Natural logarithm. Either signed zero returns negative infinity;
/// negative inputs return NaN, and positive infinity returns positive infinity.
pub fn ln(value: f32) -> f32 {
    crate::intrinsics::__metal_ln_f32(value)
}

/// Exponential with base e. Negative infinity returns positive zero;
/// positive infinity and overflow return positive infinity.
pub fn exp(value: f32) -> f32 {
    crate::intrinsics::__metal_exp_f32(value)
}

/// Sine of an angle in radians. Preserves signed zero; infinite inputs return NaN.
pub fn sin(value: f32) -> f32 {
    crate::intrinsics::__metal_sin_f32(value)
}

/// Cosine of an angle in radians. Infinite inputs return NaN.
pub fn cos(value: f32) -> f32 {
    crate::intrinsics::__metal_cos_f32(value)
}

/// Returns `(sin(value), cos(value))` from one Metal `sincos` call, with the same edge
/// cases as the separate functions; it is faster in the precise mode.
pub fn sin_cos(value: f32) -> (f32, f32) {
    crate::intrinsics::__metal_sin_cos_f32(value)
}

/// Raises `base` to a floating-point exponent. Negative finite bases with a
/// finite noninteger exponent return NaN. Any base to zero and one to any
/// exponent return one, including when the other argument is NaN.
pub fn powf(base: f32, exponent: f32) -> f32 {
    crate::intrinsics::__metal_powf_f32(base, exponent)
}

/// Hyperbolic tangent. Preserves signed zero; infinities return their signed unit.
pub fn tanh(value: f32) -> f32 {
    crate::intrinsics::__metal_tanh_f32(value)
}

/// Inverse tangent in radians. Preserves signed zero; infinities approach ±pi/2.
pub fn atan(value: f32) -> f32 {
    crate::intrinsics::__metal_atan_f32(value)
}

/// Greatest integral value no larger than the input, returned as f32.
/// Preserves signed zero and infinities; NaN returns NaN.
pub fn floor(value: f32) -> f32 {
    crate::intrinsics::__metal_floor_f32(value)
}

/// Nearest integral value, with halfway cases rounded away from zero, returned as f32.
/// Preserves signed zero and infinities; NaN returns NaN.
pub fn round(value: f32) -> f32 {
    crate::intrinsics::__metal_round_f32(value)
}

/// Least integral value no smaller than the input, returned as f32.
/// Preserves signed zero and infinities; NaN returns NaN.
pub fn ceil(value: f32) -> f32 {
    crate::intrinsics::__metal_ceil_f32(value)
}

/// Integral part of the input, rounded toward zero, returned as f32.
/// Preserves signed zero and infinities; NaN returns NaN.
pub fn trunc(value: f32) -> f32 {
    crate::intrinsics::__metal_trunc_f32(value)
}
