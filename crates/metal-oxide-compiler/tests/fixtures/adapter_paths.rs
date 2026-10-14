#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

/// Adapters whose core implementations borrow promoted constants, call constructors, read
/// uninhabited arms, divide by `NonZero` and cast pointers to pointers.
#[kernel]
pub unsafe fn adapter_paths(out: WriteBuffer<u32>, input: u32) {
    let values = [3_u32, 1, 4, 1, 5];
    let mut filtered = 0_u32;
    for value in values.iter().filter(|value| **value > 1) {
        filtered = filtered.wrapping_add(*value);
    }
    let mut stepped = 0_u32;
    for value in values.iter().step_by(2).filter(|value| **value > 3) {
        stepped = stepped.wrapping_add(*value);
    }
    let mut doubled = [1_u32, 2, 3];
    for value in doubled.iter_mut().skip(1) {
        *value *= 2;
    }
    let results = [
        values.iter().rev().all(|value| *value != input) as u32,
        (0_u32..6).take(4).count() as u32,
        values.iter().filter(|value| **value > 1).take(2).fold(0, |sum, value| sum + value),
        filtered,
        (0_u32..7).rev().step_by(3).take(3).filter(|value| *value > 1).count() as u32,
        stepped,
        doubled[1] + doubled[2],
    ];
    for (index, result) in results.iter().enumerate() {
        unsafe { out.store_unchecked(index as u32, *result) };
    }
}
