#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

fn apply<F: Fn(f32) -> f32>(function: F, value: f32) -> f32 {
    function(value)
}

/// # Safety
/// Nothing writes `values` while the kernel runs.
#[kernel]
pub unsafe fn closures(values: ReadBuffer<f32>, mut out: DisjointBuffer<f32>, factor: f32) {
    let index = thread_index();
    let values = values.as_slice();
    let i = index.get();
    let scale = |value: f32| value * factor;
    let offset = move |value: f32| value + factor;
    if let Some(out) = out.get_mut(&index)
        && i < values.len()
    {
        *out = apply(scale, values[i]) + offset(values[i]);
    }
}

fn visit<F: FnMut(f32)>(values: &[f32], mut visitor: F) {
    for value in values {
        visitor(*value);
    }
}

/// # Safety
/// Nothing writes `values` while the kernel runs.
#[kernel]
pub unsafe fn accumulate(values: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let values = values.as_slice();
    let i = index.get();
    let mut total = 0.0;
    visit(values, |value| total += value);
    let pick = |offset: usize| values[(i + offset) % values.len()];
    if let Some(out) = out.get_mut(&index)
        && i < values.len()
    {
        *out = total + pick(1);
    }
}
