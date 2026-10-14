#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

struct Pair<'a> {
    re: &'a [f32],
    im: &'a [f32],
}

fn magnitude(pair: &Pair, i: usize) -> f32 {
    pair.re[i] * pair.re[i] + pair.im[i] * pair.im[i]
}

/// # Safety
/// Nothing writes `a` or `b` while the kernel runs.
#[kernel]
pub unsafe fn tuple_slices(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let (a, b) = (a.as_slice(), b.as_slice());
    let local = [0.5_f32, 1.5, 2.5, 3.5];
    let mixed = (b, &local[..]);
    let i = index.get();
    if let Some(out) = out.get_mut(&index)
        && i < a.len()
        && i < mixed.0.len()
    {
        *out = a[i] + mixed.0[i] + mixed.1[i % 4];
    }
}

/// # Safety
/// Nothing writes `re` or `im` while the kernel runs.
#[kernel]
pub unsafe fn record_slices(re: ReadBuffer<f32>, im: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let pair = Pair {
        re: re.as_slice(),
        im: im.as_slice(),
    };
    if let Some(out) = out.get_mut(&index) {
        *out = magnitude(&pair, index.get());
    }
}

fn pick(pair: (&[f32], &[f32]), i: usize) -> f32 {
    if i < pair.0.len() && i < pair.1.len() {
        pair.0[i] + pair.1[i]
    } else {
        0.0
    }
}

/// # Safety
/// Nothing writes `a` or `b` while the kernel runs.
#[kernel]
pub unsafe fn reference_arrays(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let i = index.get();
    let slices = [a.as_slice(), b.as_slice()];
    let local = [4.0_f32, 5.0, 6.0, 7.0];
    if let Some(out) = out.get_mut(&index) {
        let chosen = slices[i % 2];
        *out = pick((slices[0], slices[1]), i) + pick((chosen, &local[..]), i % 4);
    }
}

/// # Safety
/// Nothing writes `a` while the kernel runs.
#[kernel]
pub unsafe fn indexed_tuple(a: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let context = (&index, a.as_slice());
    if let Some(out) = out.get_mut(context.0) {
        let i = context.0.get();
        if i < context.1.len() {
            *out = context.1[i] * 2.0;
        }
    }
}
