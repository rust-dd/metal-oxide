#![no_std]
use metal_oxide_device::kernel;

struct Destructor;
impl Drop for Destructor {
    fn drop(&mut self) {}
}

fn consume<T>(_: T) {}

#[kernel]
pub unsafe fn invalid() {
    consume(Destructor);
}
