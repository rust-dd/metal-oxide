#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

#[derive(Clone, Copy)]
struct Value {
    tag: u8,
    state: (u32, bool),
    samples: [f32; 2],
}

const VALUE: Value = Value { tag: 7, state: (11, true), samples: [2.5, 4.0] };

#[kernel]
pub unsafe fn aggregate_constants(out: WriteBuffer<f32>, seed: f32) {
    let mut value = VALUE;
    let tuple = (3_u32, false);
    value.state = tuple;
    value.samples[0] = seed;
    if value.state.1 {
        unsafe { out.store_unchecked(0, -1.0); }
        return;
    }
    unsafe {
        out.store_unchecked(0, value.tag as f32);
        out.store_unchecked(1, value.state.0 as f32);
        out.store_unchecked(2, value.samples[0]);
        out.store_unchecked(3, value.samples[1]);
    }
}
