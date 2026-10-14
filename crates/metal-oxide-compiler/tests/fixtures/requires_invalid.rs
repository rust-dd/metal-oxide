#![no_std]

use metal_oxide_device::{DisjointBuffer, kernel, thread_index};

#[kernel(requires(n.len() > 0))]
pub fn fill(mut out: DisjointBuffer<u32>, n: u32) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        *out = n;
    }
}
