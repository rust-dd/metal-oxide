#![no_std]

use core::ops::Add;
use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(
    instances(scale_f32(f32, 2), scale_i32(i32, 3), scale_u32(u32, 1), alias_f32(f32, 2)),
    block = (32, 1, 1),
    requires(input.len() == output.len())
)]
pub fn scale<T: Copy + Add<Output = T>, const N: usize>(
    input: ReadBuffer<T>,
    mut output: DisjointBuffer<T>,
) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let mut value = input.as_slice()[index.get()];
        for _ in 0..N {
            value = value + value;
        }
        *output = value;
    }
}

#[kernel(instances(disabled(u32)))]
#[cfg(any())]
pub fn omitted<T: Copy>(_input: ReadBuffer<T>) {}

#[kernel(instances(conditionally_disabled(u32)))]
#[cfg_attr(all(), cfg(any()))]
pub fn conditionally_omitted<T: Copy>(_input: ReadBuffer<T>) {}
