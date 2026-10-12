#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

trait Mode {
    const SCALED: bool;
    const LANES: u32;
}

struct Plain;

impl Mode for Plain {
    const SCALED: bool = false;
    const LANES: u32 = 2;
}

fn shifted<M: Mode>(value: f32) -> f32 {
    let mut result = value;
    if M::SCALED {
        result *= 1234.5;
    }
    match M::LANES {
        1 => result += 3.0,
        2 => result += 5.0,
        _ => result += 7.0,
    }
    result
}

#[kernel]
pub unsafe fn constant_branches(out: WriteBuffer<f32>, value: f32) {
    unsafe { out.store_unchecked(0, shifted::<Plain>(value)) };
}
