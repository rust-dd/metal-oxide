/// Square root with Metal's precise f32 math mode.
/// Subnormal inputs and outputs may be flushed to zero by the GPU.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_sqrt")]
pub fn sqrt(_value: f32) -> f32 {
    panic!("sqrt is only available in Metal kernels")
}

/// Computes `a * b + c` with one rounding, without rounding the product first.
/// Subnormal inputs and outputs may be flushed to zero by the GPU.
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_fma")]
pub fn fma(_a: f32, _b: f32, _c: f32) -> f32 {
    panic!("fma is only available in Metal kernels")
}
