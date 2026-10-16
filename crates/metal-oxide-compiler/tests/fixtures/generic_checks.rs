#![no_std]

use metal_oxide_device::{DisjointBuffer, kernel, thread_index};

#[kernel(instances(divide_two(2), divide_zero(0)))]
pub fn divide<const DIVISOR: u32>(mut output: DisjointBuffer<u32>, value: u32) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        *output = value / DIVISOR;
    }
}
