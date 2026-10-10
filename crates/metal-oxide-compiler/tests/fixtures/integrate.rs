#![no_std]

use metal_oxide_device::{ReadWriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[kernel]
pub unsafe fn integrate(values: ReadWriteBuffer<f32>, rows: u32, steps: u32, dt: f32) {
    let row = block_idx().x * block_dim().x + thread_idx().x;
    if row < rows {
        let mut state = 1.0;
        let mut step = 0;
        while step < steps {
            let index = row * steps + step;
            unsafe {
                state = state + 0.25 * state * dt + values.load_unchecked(index);
                values.store_unchecked(index, state);
            }
            step += 1;
        }
    }
}
