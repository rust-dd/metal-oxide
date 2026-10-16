#![no_std]

use metal_oxide_device::{ReadBuffer, kernel};

#[cfg(empty)]
#[kernel(instances())]
pub fn empty<T: Copy>(_input: ReadBuffer<T>) {}

#[cfg(duplicate)]
#[kernel(instances(same(f32), same(u32)))]
pub fn duplicate<T: Copy>(_input: ReadBuffer<T>) {}

#[cfg(nongeneric)]
#[kernel(instances(concrete(u32)))]
pub fn nongeneric(_input: ReadBuffer<u32>) {}

#[cfg(arity)]
#[kernel(instances(missing(u32)))]
pub fn arity<T: Copy, const N: u32>(_input: ReadBuffer<T>) {}

#[cfg(bound)]
#[kernel(instances(invalid(f32)))]
pub fn bound<T: Copy + core::ops::BitAnd<Output = T>>(_input: ReadBuffer<T>) {}

#[cfg(unsupported)]
#[kernel(instances(unsupported_type(f64)))]
pub fn unsupported<T: Copy>(_input: ReadBuffer<T>) {}

#[cfg(collision)]
#[kernel(instances(colliding(f32)))]
pub fn collision<T: Copy>(_input: ReadBuffer<T>) {}

#[cfg(collision)]
#[kernel]
pub fn colliding(_input: ReadBuffer<f32>) {}

#[cfg(calls_entry)]
#[kernel(instances(callee_u32(u32)))]
pub fn callee<T: Copy>(_input: ReadBuffer<T>) {}

#[cfg(calls_entry)]
#[kernel]
pub fn caller(input: ReadBuffer<u32>) { callee::<u32>(input); }

#[cfg(malformed_marker)]
#[metal_oxide_compiler::instance]
const _: () = ();

#[cfg(unknown_template)]
#[metal_oxide_compiler::instance = "unknown"]
const _: () = { let _ = not_a_kernel::<u32>; };

#[cfg(unknown_template)]
fn not_a_kernel<T>() {}
