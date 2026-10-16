#![no_std]
#![feature(portable_simd)]

use core::simd::{Mask, Simd};
use metal_oxide_device::{WriteBuffer, kernel};

#[kernel]
pub unsafe fn rejected(output: WriteBuffer<f32>, value: f32) {
    let mut values = [value; 4];
    let lanes = Simd::from_array(values);
    #[cfg(gather)]
    let result = Simd::gather_or(&values, Simd::from_array([0, 1, 2, 3]), lanes);
    #[cfg(scatter)]
    let result = {
        lanes.scatter(&mut values, Simd::from_array([0, 1, 2, 3]));
        Simd::from_array(values)
    };
    #[cfg(masked_load)]
    let result = Simd::load_select(&values, Mask::from_array([true, false, true, false]), lanes);
    #[cfg(masked_store)]
    let result = {
        lanes.store_select(&mut values, Mask::from_array([true, false, true, false]));
        Simd::from_array(values)
    };
    #[cfg(masked_load_ptr)]
    let result = unsafe {
        Simd::load_select_ptr(
            values.as_ptr(),
            Mask::from_array([true, false, true, false]),
            lanes,
        )
    };
    #[cfg(masked_store_ptr)]
    let result = {
        unsafe {
            lanes.store_select_ptr(
                values.as_mut_ptr(),
                Mask::from_array([true, false, true, false]),
            );
        }
        Simd::from_array(values)
    };
    // SAFETY: the caller supplies one output element and launches one thread.
    unsafe { output.store_unchecked(0, result[0]) };
}
