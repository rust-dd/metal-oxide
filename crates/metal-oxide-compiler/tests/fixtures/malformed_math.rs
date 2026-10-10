#![no_std]

pub use metal_oxide_macros::kernel;

pub mod intrinsics {
    unsafe extern "Rust" {
        #[cfg(unary)]
        pub safe fn __metal_ln_f32(value: u32) -> u32;
        #[cfg(binary)]
        pub safe fn __metal_powf_f32(base: f32, exponent: u32) -> f32;
        #[cfg(arity)]
        pub safe fn __metal_powf_f32(base: f32) -> f32;
    }
}
