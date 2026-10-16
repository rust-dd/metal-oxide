#![no_std]
#![feature(f16)]
use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn negate_bits(input: ReadBuffer<u32>, mut output: DisjointBuffer<[u32; 2]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let bits = input.as_slice()[index.get()];
        *output = [(-f32::from_bits(bits)).to_bits(), (-f16::from_bits(bits as u16)).to_bits() as u32];
    }
}
