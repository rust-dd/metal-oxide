#![no_std]
use metal_oxide_device::{WriteBuffer, kernel, threadgroup};

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>, a: u32, b: u32, c: u32) {
    'no_barrier: {
        'skip_store: {
            loop {
                if a == 0 {
                    break 'skip_store;
                }
                if b == 0 {
                    break;
                }
                if c == 0 {
                    break 'no_barrier;
                }
            }
            unsafe { output.store_unchecked(0, 1) };
        }
        unsafe { threadgroup::barrier() };
    }
}
