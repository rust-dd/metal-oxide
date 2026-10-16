#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

struct View<'a, T> {
    values: &'a [T],
}

impl<'a, T> View<'a, T> {
    fn get(&self, index: usize) -> &'a T {
        &self.values[index]
    }
}

fn identity<'a, T>(value: &'a T) -> &'a T {
    value
}

fn apply<T, F: for<'a> Fn(&'a T) -> &'a T>(value: &T, f: F) -> &T {
    f(value)
}

fn reborrow<'a, 'b: 'a>(value: &'a mut &'b mut u32) -> &'a mut u32 {
    value
}

#[kernel(requires(input.len() == output.len()))]
pub fn late<'a>(input: ReadBuffer<u32>, mut output: DisjointBuffer<u32>) {
    let index = thread_index();
    let view = View { values: input.as_slice() };
    if let Some(value) = output.get_mut(&index) {
        *reborrow(&mut { value }) = *apply(view.get(index.get()), identity);
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn early<'a: 'b, 'b>(input: ReadBuffer<u32>, mut output: DisjointBuffer<u32>) {
    let index = thread_index();
    if let Some(value) = output.get_mut(&index) {
        *value = *View { values: input.as_slice() }.get(index.get());
    }
}

#[kernel(instances(generic_u32(u32)), requires(input.len() == output.len()))]
pub fn generic<'a, T: Copy + 'a>(input: ReadBuffer<T>, mut output: DisjointBuffer<T>) {
    let index = thread_index();
    if let Some(value) = output.get_mut(&index) {
        *value = *apply(&input.as_slice()[index.get()], identity);
    }
}
