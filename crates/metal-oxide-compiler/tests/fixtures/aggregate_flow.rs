#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

#[derive(Clone, Copy)]
struct State { point: (f32, f32), weights: [f32; 2] }

fn update(mut value: State, mode: u32) -> State {
    let original = value;
    if mode == 0 { return original; }
    match mode {
        1 => value.point.0 = value.weights[0],
        2 => value.point.0 = original.point.1,
        _ => value.point.0 = 7.0,
    }
    let mut step = 0;
    while step < mode {
        if step == 1 { step += 1; continue; }
        value.weights[1] += original.point.0;
        if value.weights[1] > 50.0 { break; }
        step += 1;
    }
    value
}

#[kernel]
pub unsafe fn aggregate_flow(out: WriteBuffer<f32>, seed: f32, mode: u32) {
    let value = State { point: (seed, seed + 1.0), weights: [seed + 2.0, seed + 3.0] };
    let updated = update(value, mode);
    unsafe {
        out.store_unchecked(0, updated.point.0);
        out.store_unchecked(1, updated.point.1);
        out.store_unchecked(2, updated.weights[0]);
        out.store_unchecked(3, updated.weights[1]);
        out.store_unchecked(4, value.weights[1]);
    }
}
