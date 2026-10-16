#![no_std]
#![feature(portable_simd)]

use core::cmp::Ordering;
use core::simd::{Mask, Simd};
use metal_oxide_device::{ReadBuffer, kernel};

#[derive(Clone, Copy)]
pub struct Nested {
    vectors: [(Simd<f32, 4>, u32); 2],
}

#[cfg(direct)]
type Parameter = Simd<f32, 4>;
#[cfg(mask)]
type Parameter = Mask<i32, 4>;
#[cfg(buffer)]
type Parameter = ReadBuffer<Simd<f32, 4>>;
#[cfg(nested)]
type Parameter = Nested;
#[cfg(nested_buffer)]
type Parameter = ReadBuffer<Nested>;
#[cfg(ordering)]
type Parameter = Ordering;
#[cfg(ordering_buffer)]
type Parameter = ReadBuffer<Ordering>;

#[kernel]
pub fn rejected(_value: Parameter) {}
