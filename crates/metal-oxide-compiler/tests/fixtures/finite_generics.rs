#![no_std]
use metal_oxide_device::kernel;

trait Step {
    fn step(self);
}
fn forward<T: Step>(value: T) {
    value.step();
}
impl Step for u32 {
    fn step(self) {
        forward(self as f32);
    }
}
impl Step for f32 {
    fn step(self) {}
}

#[kernel]
pub unsafe fn concrete(value: u32) {
    forward(value);
}
