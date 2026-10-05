#![no_std]
use metal_oxide_device::kernel;

struct Destructor;
impl Drop for Destructor {
    fn drop(&mut self) {}
}

#[kernel]
pub unsafe fn invalid() {
    let _value = Destructor;
}
