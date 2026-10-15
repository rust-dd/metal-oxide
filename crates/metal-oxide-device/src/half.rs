/// IEEE binary16 storage. Convert to f32 before arithmetic.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct F16(u16);

impl F16 {
    /// Constructs a value from its exact binary16 representation.
    pub const fn from_bits(bits: u16) -> Self {
        // SAFETY: F16 is transparent over u16 and every bit pattern is valid.
        unsafe { core::mem::transmute::<u16, Self>(bits) }
    }

    /// Returns the exact binary16 representation.
    pub const fn to_bits(self) -> u16 {
        // SAFETY: F16 is transparent over u16 and every bit pattern is valid.
        unsafe { core::mem::transmute::<Self, u16>(self) }
    }

    /// Rounds f32 to binary16 using round to nearest, ties to even.
    pub fn from_f32(value: f32) -> Self {
        crate::intrinsics::__metal_f16_from_f32(value)
    }

    /// Widens binary16 to f32 without rounding finite values.
    pub fn to_f32(self) -> f32 {
        crate::intrinsics::__metal_f16_to_f32(self)
    }
}

#[cfg(test)]
mod tests;
