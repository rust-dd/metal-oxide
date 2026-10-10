use metal_oxide_device::{ReadWriteBuffer, kernel};

/// # Safety
/// values covers rows * steps initialized increments. Each active lane owns a row.
#[kernel]
pub unsafe fn integrate(values: ReadWriteBuffer<f32>, rows: u32, steps: u32, dt: f32) {
    let row = crate::index();
    if row < rows {
        let mut state = 1.0;
        let mut step = 0;
        while step < steps {
            let index = row * steps + step;
            // SAFETY: each active row owns steps initialized values.
            unsafe {
                state = state + 0.25 * state * dt + values.load_unchecked(index);
                values.store_unchecked(index, state);
            }
            step += 1;
        }
    }
}
