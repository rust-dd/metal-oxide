#![no_std]
#![feature(pattern_types, pattern_type_macro)]
use metal_oxide_device::kernel;

type Positive = core::pattern_type!(u32 is 1..);

#[kernel]
pub unsafe fn invalid(value: u32) {
    let positive = unsafe { core::mem::transmute::<u32, Positive>(value | 1) };
    // Rustc resolves no instance for the built-in `Clone` of a pattern type.
    let _ = positive.clone();
}
