#![no_std]
#![feature(portable_simd)]

use core::simd::Simd;
use metal_oxide_device::{WriteBuffer, kernel};

#[cfg(float64)]
type Element = f64;
#[cfg(signed64)]
type Element = i64;
#[cfg(unsigned64)]
type Element = u64;
#[cfg(not(any(float64, signed64, unsigned64)))]
type Element = f32;

#[cfg(wide)]
const LANES: usize = 8;
#[cfg(single)]
const LANES: usize = 1;
#[cfg(not(any(wide, single)))]
const LANES: usize = 4;

#[kernel]
pub unsafe fn rejected(output: WriteBuffer<u32>) {
    let value = Simd::<Element, LANES>::splat(1 as Element);
    // SAFETY: the caller supplies one output element and launches one thread.
    unsafe { output.store_unchecked(0, value[0] as u32) };
}
