/// IEEE binary16 storage. Convert to f32 before arithmetic.
#[repr(transparent)]
#[derive(Clone, Copy)]
#[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_f16")]
pub struct F16(u16);

impl F16 {
    /// Constructs a value from its exact binary16 representation.
    #[cfg_attr(
        target_env = "metal",
        rustc_diagnostic_item = "metal_oxide_f16_from_bits"
    )]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the exact binary16 representation.
    #[cfg_attr(
        target_env = "metal",
        rustc_diagnostic_item = "metal_oxide_f16_to_bits"
    )]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Rounds f32 to binary16 using round to nearest, ties to even.
    #[cfg_attr(
        target_env = "metal",
        rustc_diagnostic_item = "metal_oxide_f16_from_f32"
    )]
    pub fn from_f32(_value: f32) -> Self {
        panic!("F16 conversion is only available in Metal kernels")
    }

    /// Widens binary16 to f32 without rounding finite values.
    #[cfg_attr(target_env = "metal", rustc_diagnostic_item = "metal_oxide_f16_to_f32")]
    pub fn to_f32(self) -> f32 {
        panic!("F16 conversion is only available in Metal kernels")
    }
}
