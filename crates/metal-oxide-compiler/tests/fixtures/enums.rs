#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

enum Shape {
    Circle(f32),
    Square(f32),
    Rect(f32, f32),
    Empty,
}

fn shape(kind: u32, size: f32) -> Shape {
    match kind % 4 {
        0 => Shape::Circle(size),
        1 => Shape::Square(size),
        2 => Shape::Rect(size, size + 1.0),
        _ => Shape::Empty,
    }
}

fn area(shape: Shape) -> f32 {
    match shape {
        Shape::Circle(radius) => 3.0 * radius * radius,
        Shape::Square(side) => side * side,
        Shape::Rect(width, height) => width * height,
        Shape::Empty => 0.0,
    }
}

/// # Safety
/// Nothing writes `sizes` while the kernel runs.
#[kernel]
pub unsafe fn areas(sizes: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let sizes = sizes.as_slice();
    let i = index.get();
    if let Some(out) = out.get_mut(&index)
        && i < sizes.len()
    {
        *out = area(shape(i as u32, sizes[i]));
    }
}

fn largest(values: &[f32]) -> Option<&f32> {
    let mut best: Option<&f32> = None;
    for value in values {
        best = match best {
            Some(current) if *current >= *value => Some(current),
            _ => Some(value),
        };
    }
    best
}

/// # Safety
/// Nothing writes `values` while the kernel runs.
#[kernel]
pub unsafe fn largest_prefix(values: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    let values = values.as_slice();
    let i = index.get();
    if let Some(out) = out.get_mut(&index)
        && i < values.len()
    {
        *out = match largest(&values[..i + 1]) {
            Some(value) => *value,
            None => -1.0,
        };
    }
}
