/// Square root with Metal's precise f32 math mode.
/// Subnormal inputs and outputs may be flushed to zero by the GPU.
pub fn sqrt(value: f32) -> f32 {
    crate::intrinsics::__metal_sqrt_f32(value)
}

/// Computes `a * b + c` with one rounding, without rounding the product first.
/// Subnormal inputs and outputs may be flushed to zero by the GPU.
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    crate::intrinsics::__metal_fma_f32(a, b, c)
}
