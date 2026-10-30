use crate::{BF16, F16, intrinsics};

#[cfg(test)]
mod tests;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for crate::F16 {}
    impl Sealed for crate::BF16 {}
}

/// Math methods with the names and signatures of Rust's `f32` methods, which `core` lacks.
/// Kernels in host crates have it in scope; separate kernel crates import it.
pub trait Float: Copy + sealed::Sealed {
    /// Square root; negative inputs return NaN.
    fn sqrt(self) -> Self;
    /// `self * a + b` with one rounding.
    fn mul_add(self, a: Self, b: Self) -> Self;
    /// Natural logarithm; zero returns negative infinity and negative inputs NaN.
    fn ln(self) -> Self;
    /// Base-2 logarithm; zero returns negative infinity and negative inputs NaN.
    fn log2(self) -> Self;
    /// Base-10 logarithm; zero returns negative infinity and negative inputs NaN.
    fn log10(self) -> Self;
    /// `e` raised to `self`.
    fn exp(self) -> Self;
    /// 2 raised to `self`.
    fn exp2(self) -> Self;
    /// Sine of an angle in radians.
    fn sin(self) -> Self;
    /// Cosine of an angle in radians.
    fn cos(self) -> Self;
    /// Tangent of an angle in radians.
    fn tan(self) -> Self;
    /// `(sin, cos)` from one Metal `sincos` call, which is faster than two in the precise mode.
    fn sin_cos(self) -> (Self, Self);
    /// Arcsine in radians; inputs outside `[-1, 1]` return NaN.
    fn asin(self) -> Self;
    /// Arccosine in radians; inputs outside `[-1, 1]` return NaN.
    fn acos(self) -> Self;
    /// Arctangent in radians.
    fn atan(self) -> Self;
    /// The angle of the point `(other, self)` in radians, in `[-pi, pi]`.
    fn atan2(self, other: Self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// `self` raised to the power `n`.
    fn powf(self, n: Self) -> Self;
    /// `self` raised to the integer power `n` by repeated squaring, as compiler-rt computes it.
    fn powi(self, n: i32) -> Self;
    /// The length of the hypotenuse with legs `self` and `other`, without intermediate
    /// overflow; an infinite leg gives infinity even when the other one is NaN.
    fn hypot(self, other: Self) -> Self;
    /// `1 / sqrt(self)` from one Metal `rsqrt` call.
    fn rsqrt(self) -> Self;
    /// The largest integer no greater than `self`.
    fn floor(self) -> Self;
    /// The smallest integer no less than `self`.
    fn ceil(self) -> Self;
    /// The nearest integer, with halfway cases rounded away from zero.
    fn round(self) -> Self;
    /// The integer part of `self`, rounded toward zero.
    fn trunc(self) -> Self;
}

impl Float for f32 {
    fn sqrt(self) -> f32 {
        intrinsics::__metal_sqrt_f32(self)
    }

    fn mul_add(self, a: f32, b: f32) -> f32 {
        intrinsics::__metal_fma_f32(self, a, b)
    }

    fn ln(self) -> f32 {
        intrinsics::__metal_ln_f32(self)
    }

    fn log2(self) -> f32 {
        intrinsics::__metal_log2_f32(self)
    }

    fn log10(self) -> f32 {
        intrinsics::__metal_log10_f32(self)
    }

    fn exp(self) -> f32 {
        intrinsics::__metal_exp_f32(self)
    }

    fn exp2(self) -> f32 {
        intrinsics::__metal_exp2_f32(self)
    }

    fn sin(self) -> f32 {
        intrinsics::__metal_sin_f32(self)
    }

    fn cos(self) -> f32 {
        intrinsics::__metal_cos_f32(self)
    }

    fn tan(self) -> f32 {
        intrinsics::__metal_tan_f32(self)
    }

    fn sin_cos(self) -> (f32, f32) {
        intrinsics::__metal_sin_cos_f32(self)
    }

    fn asin(self) -> f32 {
        intrinsics::__metal_asin_f32(self)
    }

    fn acos(self) -> f32 {
        intrinsics::__metal_acos_f32(self)
    }

    fn atan(self) -> f32 {
        intrinsics::__metal_atan_f32(self)
    }

    fn atan2(self, other: f32) -> f32 {
        intrinsics::__metal_atan2_f32(self, other)
    }

    fn tanh(self) -> f32 {
        intrinsics::__metal_tanh_f32(self)
    }

    fn powf(self, n: f32) -> f32 {
        intrinsics::__metal_powf_f32(self, n)
    }

    fn powi(self, n: i32) -> f32 {
        let mut base = self;
        let mut exponent = n.unsigned_abs();
        let mut result = 1.0;
        loop {
            if exponent & 1 != 0 {
                result *= base;
            }
            exponent >>= 1;
            if exponent == 0 {
                break;
            }
            base *= base;
        }
        if n < 0 { 1.0 / result } else { result }
    }

    fn hypot(self, other: f32) -> f32 {
        let x = self.abs();
        let y = other.abs();
        if x == f32::INFINITY || y == f32::INFINITY {
            return f32::INFINITY;
        }
        if x.is_nan() || y.is_nan() {
            return f32::NAN;
        }
        let (big, small) = if x < y { (y, x) } else { (x, y) };
        if big == 0.0 {
            return 0.0;
        }
        let ratio = small / big;
        big * (1.0 + ratio * ratio).sqrt()
    }

    fn rsqrt(self) -> f32 {
        intrinsics::__metal_rsqrt_f32(self)
    }

    fn floor(self) -> f32 {
        intrinsics::__metal_floor_f32(self)
    }

    fn ceil(self) -> f32 {
        intrinsics::__metal_ceil_f32(self)
    }

    fn round(self) -> f32 {
        intrinsics::__metal_round_f32(self)
    }

    fn trunc(self) -> f32 {
        intrinsics::__metal_trunc_f32(self)
    }
}

/// Implements [`Float`] for a 16-bit type by computing in f32 and rounding the result once.
macro_rules! through_f32 {
    ($ty:ty, $($unary:ident),*) => {
        impl Float for $ty {
            $(fn $unary(self) -> Self {
                Self::from_f32(self.to_f32().$unary())
            })*

            fn mul_add(self, a: Self, b: Self) -> Self {
                Self::from_f32(self.to_f32().mul_add(a.to_f32(), b.to_f32()))
            }

            fn sin_cos(self) -> (Self, Self) {
                let (sine, cosine) = self.to_f32().sin_cos();
                (Self::from_f32(sine), Self::from_f32(cosine))
            }

            fn atan2(self, other: Self) -> Self {
                Self::from_f32(self.to_f32().atan2(other.to_f32()))
            }

            fn powf(self, n: Self) -> Self {
                Self::from_f32(self.to_f32().powf(n.to_f32()))
            }

            fn powi(self, n: i32) -> Self {
                Self::from_f32(self.to_f32().powi(n))
            }

            fn hypot(self, other: Self) -> Self {
                Self::from_f32(self.to_f32().hypot(other.to_f32()))
            }
        }
    };
}

through_f32!(
    F16, sqrt, ln, log2, log10, exp, exp2, sin, cos, tan, asin, acos, atan, tanh, rsqrt, floor,
    ceil, round, trunc
);
through_f32!(
    BF16, sqrt, ln, log2, log10, exp, exp2, sin, cos, tan, asin, acos, atan, tanh, rsqrt, floor,
    ceil, round, trunc
);
