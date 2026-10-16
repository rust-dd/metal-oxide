/// Brain floating point, with eight exponent bits and seven fraction bits.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct BF16(u16);

impl BF16 {
    /// Constructs a value from its exact bfloat16 representation.
    pub const fn from_bits(bits: u16) -> Self {
        // SAFETY: BF16 is transparent over u16 and every bit pattern is valid.
        unsafe { core::mem::transmute::<u16, Self>(bits) }
    }

    /// Returns the exact bfloat16 representation.
    pub const fn to_bits(self) -> u16 {
        // SAFETY: BF16 is transparent over u16 and every bit pattern is valid.
        unsafe { core::mem::transmute::<Self, u16>(self) }
    }

    /// Rounds f32 to bfloat16 using round to nearest, ties to even.
    pub fn from_f32(value: f32) -> Self {
        crate::intrinsics::__metal_bf16_from_f32(value)
    }

    /// Widens bfloat16 to f32 without rounding finite values.
    pub fn to_f32(self) -> f32 {
        crate::intrinsics::__metal_bf16_to_f32(self)
    }
}

mod ops;
