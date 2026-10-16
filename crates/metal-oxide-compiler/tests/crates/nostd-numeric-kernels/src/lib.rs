#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};
use num_traits::{Float, PrimInt};

/// Computes bitwise, decomposition and rounding operations through libm's Rust implementation.
pub fn libm_values(value: f32) -> [f32; 8] {
    let (fraction, integer) = libm::modff(value);
    [
        libm::fabsf(value),
        libm::copysignf(1.0, value),
        fraction,
        integer,
        libm::floorf(value),
        libm::ceilf(value),
        libm::roundf(value),
        libm::truncf(value),
    ]
}

/// Exercises the no_std Float implementation, including its generic integer-power algorithm.
pub fn float_values<T: Float>(value: T, exponent: i32) -> [T; 4] {
    [
        value.floor(),
        value.ceil(),
        value.abs(),
        value.powi(exponent),
    ]
}

/// Computes checked powers, rotations and saturation through generic numeric traits.
pub fn integer_values<T: PrimInt>(value: T, exponent: u32) -> [T; 4] {
    [
        num_traits::checked_pow(value, exponent as usize).unwrap_or_else(T::max_value),
        value.rotate_left(exponent),
        value.saturating_add(T::one()),
        value.saturating_sub(T::one()),
    ]
}

/// Computes libm's supported f32 helpers for each input element.
#[kernel(requires(input.len() == output.len()))]
pub fn libm_f32(input: ReadBuffer<f32>, mut output: DisjointBuffer<[f32; 8]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = libm_values(input.as_slice()[index.get()]);
    }
}

/// Computes a concrete Float workload through the dependency's MIR.
#[kernel(instances(traits_f32(f32)), requires(input.len() == output.len()))]
pub fn float<T: Float>(input: ReadBuffer<T>, mut output: DisjointBuffer<[T; 4]>, exponent: i32) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = float_values(input.as_slice()[index.get()], exponent);
    }
}

/// Computes concrete integer workloads through the dependency's MIR.
#[kernel(
    instances(
        traits_i8(i8), traits_i16(i16), traits_i32(i32),
        traits_u8(u8), traits_u16(u16), traits_u32(u32), traits_u64(u64)
    ),
    requires(input.len() == output.len())
)]
pub fn integer<T: PrimInt>(
    input: ReadBuffer<T>,
    mut output: DisjointBuffer<[T; 4]>,
    exponent: u32,
) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = integer_values(input.as_slice()[index.get()], exponent);
    }
}

/// This f32 entry reaches libm's unsupported f64 intermediate calculations.
#[cfg(feature = "unsupported-f64")]
#[kernel(requires(input.len() == output.len()))]
pub fn libm_cube_root(input: ReadBuffer<f32>, mut output: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = libm::cbrtf(input.as_slice()[index.get()]);
    }
}

/// This entry reaches unsupported volatile and foreign calls in libm's argument reduction.
#[cfg(feature = "unsupported-sine")]
#[kernel(requires(input.len() == output.len()))]
pub fn libm_sine(input: ReadBuffer<f32>, mut output: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = libm::sinf(input.as_slice()[index.get()]);
    }
}
