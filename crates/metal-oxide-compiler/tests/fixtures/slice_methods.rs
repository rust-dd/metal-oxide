#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn rearranged(input: ReadBuffer<([u32; 8], u32, u32)>, mut output: DisjointBuffer<[u32; 8]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let (values, i, j) = input.as_slice()[index.get()];
        *output = values;
        output.swap(i as usize % 8, j as usize % 8);
        output[1..7].reverse();
        let mut local = *output;
        local[..i as usize % 9].reverse();
        let (head, tail) = local.split_at_mut(4);
        core::mem::swap(&mut head[j as usize % 4], &mut tail[3]);
        let same = &raw mut local[i as usize % 8];
        unsafe { core::ptr::swap(same, same) };
        *output = local;
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn copied(input: ReadBuffer<([u32; 8], u32, u32)>, mut output: DisjointBuffer<[u32; 8]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let (values, a, b) = input.as_slice()[index.get()];
        let mut local = [7; 8];
        local[..4].copy_from_slice(&values[4..]);
        output.copy_from_slice(&local);
        let tail = &input.as_slice()[index.get()].0[b as usize % 9..];
        output[a as usize % 9..].copy_from_slice(tail);
    }
}
