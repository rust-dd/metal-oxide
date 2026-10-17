#![no_std]

use metal_oxide_device::{buffer::ReadBuffer, kernel, simdgroup::Matrix};

#[kernel(block = (32, 1, 1))]
pub unsafe fn invalid_matrix(input: ReadBuffer<f32>) {
    #[cfg(safe_load)]
    let _ = input.load_matrix(0, 0, 8);
    #[cfg(other_element)]
    let _ = unsafe { input.load_matrix(0, 0, 8) };
    #[cfg(read_store)]
    unsafe { input.store_matrix(0, 0, 8, Matrix { lanes: [0.0; 2] }) };
}
