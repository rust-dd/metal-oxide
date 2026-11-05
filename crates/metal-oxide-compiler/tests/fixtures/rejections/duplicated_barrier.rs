#![no_std]
use metal_oxide_device::{WriteBuffer, kernel, threadgroup};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, a: u32) {
    'skip_barrier: {
        match a {
            0 => unsafe { output.store_unchecked(0, 1) },
            1 => unsafe { output.store_unchecked(0, 2) },
            _ => break 'skip_barrier,
        }
        unsafe { threadgroup::barrier() };
    }
    unsafe { output.store_unchecked(1, 3) };
}
