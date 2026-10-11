#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel, uninit_array};

#[kernel]
pub unsafe fn running_sums(out: WriteBuffer<f32>, input: ReadBuffer<f32>, count: u32) {
    let count = if count < 64 { count } else { 64 };
    // SAFETY: element k is written at step k before any read of index k.
    let mut history: [f32; 64] = unsafe { uninit_array() };
    let mut step = 0u32;
    while step < count {
        history[step as usize] = unsafe { input.load_unchecked(step) };
        let mut sum = 0.0;
        let mut position = step;
        loop {
            sum += history[position as usize];
            if position == 0 {
                break;
            }
            position -= 1;
        }
        unsafe { out.store_unchecked(step, sum) };
        step += 1;
    }
}
