#![no_std]

use metal_oxide_device::{WriteBuffer, kernel, thread_idx};

/// # Safety
/// `output` holds one value per thread.
#[kernel]
pub unsafe fn option_value(output: WriteBuffer<u32>, n: u32) {
    let value = if n > 3 { Some(n) } else { None };
    if let Some(value) = value {
        unsafe { output.store_unchecked(thread_idx().x, value) };
    }
}
