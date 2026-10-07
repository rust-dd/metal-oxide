#![no_std]

pub use metal_oxide_macros::kernel;

pub mod intrinsics {
    unsafe extern "Rust" {
        pub safe fn __metal_sqrt_f32(value: u32) -> u32;
    }
}
