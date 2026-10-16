#![no_std]

use metal_oxide_device::{DisjointBuffer, kernel, thread_index};

#[cfg(escape)]
fn invalid<'a>() -> &'a u32 {
    let value = 7;
    &value
}

#[cfg(outlives)]
fn invalid<'a, 'b>(value: &'b u32) -> &'a u32 {
    value
}

#[kernel]
pub fn entry(mut output: DisjointBuffer<u32>) {
    let index = thread_index();
    if let Some(value) = output.get_mut(&index) {
        #[cfg(escape)]
        { *value = *invalid(); }
        #[cfg(outlives)]
        { *value = *invalid(&7); }
    }
}
