#![no_std]

use metal_oxide_device::{buffer::ReadBuffer, kernel, simdgroup::Matrix};

#[kernel(block = (32, 1, 1))]
pub unsafe fn invalid_matrix(input: ReadBuffer<f32>) {
    #[cfg(safe_load)]
    let _ = input.load_matrix(0, 0, 8);
    #[cfg(any(other_element, bridge_flag, two_calls))]
    let _ = unsafe { input.load_matrix(0, 0, 8) };
    #[cfg(read_store)]
    unsafe { input.store_matrix(0, 0, 8, Matrix { lanes: [0.0; 2] }) };
}

#[cfg(wrong_buffer)]
#[kernel(block = (32, 1, 1))]
pub unsafe fn wrong_buffer(output: metal_oxide_device::buffer::WriteBuffer<f32>) {
    let _ = unsafe { ReadBuffer::load_matrix(output, 0, 0, 8) };
}
