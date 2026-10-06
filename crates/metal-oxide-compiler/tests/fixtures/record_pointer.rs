#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};
struct Pointer { value: &'static f32 }
#[kernel]
pub unsafe fn pointer(output: WriteBuffer<f32>) {
    let record = Pointer { value: &1.0 };
    // SAFETY: one thread owns output[0].
    unsafe { output.store_unchecked(0,*record.value); }
}
