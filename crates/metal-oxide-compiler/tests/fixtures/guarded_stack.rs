#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel, thread_idx};

#[kernel]
pub unsafe fn guarded_stack(input: ReadBuffer<u32>, output: WriteBuffer<f32>, count: u32) {
    let mut stack = [0.0; 8];
    let mut i = 0;
    while i < count {
        let at = i / 100;
        if at < 8 {
            let choice = unsafe { input.load_unchecked(i) };
            stack[at as usize] = if choice == 0 { 1.0 } else if choice == 1 { 2.0 } else { 3.0 };
        }
        i += 1;
    }
    unsafe { output.store_unchecked(thread_idx().x, stack[0]) };
}
