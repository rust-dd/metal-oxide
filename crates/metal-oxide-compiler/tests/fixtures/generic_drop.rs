#![no_std]
use metal_oxide_device::kernel;

fn consume<T>(_: T) {}

#[kernel]
pub unsafe fn scalar(value: u32) {
    consume(value);
}
