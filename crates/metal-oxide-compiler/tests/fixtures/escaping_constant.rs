#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

fn constant() -> &'static f32 {
    &1.0
}

#[kernel]
pub unsafe fn escaping(output: WriteBuffer<f32>) {
    // SAFETY: one thread owns output[0].
    unsafe { output.store_unchecked(0, *constant()) };
}
