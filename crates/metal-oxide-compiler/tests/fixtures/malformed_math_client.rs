#![no_std]

use metal_oxide_device::{intrinsics, kernel};

#[kernel]
pub unsafe fn invalid_math_signature() {
    #[cfg(unary)]
    let _ = intrinsics::__metal_ln_f32(1);
    #[cfg(binary)]
    let _ = intrinsics::__metal_powf_f32(2.0, 3);
    #[cfg(arity)]
    let _ = intrinsics::__metal_powf_f32(2.0);
}
