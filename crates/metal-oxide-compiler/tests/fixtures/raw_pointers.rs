#![no_std]

use core::marker::PhantomData;

use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

struct Meters;

struct Tagged<T> {
    value: f32,
    _unit: PhantomData<T>,
}

fn walk(values: &[f32]) -> f32 {
    let start = values.as_ptr();
    let end = unsafe { start.add(values.len()) };
    let mut cursor = start;
    let mut total = 0.0;
    while cursor != end {
        total += unsafe { *cursor };
        cursor = unsafe { cursor.add(1) };
    }
    total * unsafe { end.offset_from_unsigned(start) } as f32
}

fn scale(values: &mut [f32], tagged: Tagged<Meters>) {
    let data = values.as_mut_ptr();
    for i in 0..values.len() {
        unsafe { *data.add(i) *= tagged.value };
    }
}

/// # Safety
/// `out` holds four values; one thread writes them. Nothing writes `values`.
#[kernel]
pub unsafe fn raw_pointers(values: ReadBuffer<f32>, out: WriteBuffer<f32>, shift: i32) {
    let values = values.as_slice();
    let mut copy = [values[0], values[1], values[2], values[3]];
    scale(
        &mut copy,
        Tagged {
            value: 2.0,
            _unit: PhantomData,
        },
    );
    let last = unsafe { *values.as_ptr().add(values.len() - 1) };
    let offset = values.len() as isize - shift as isize;
    unsafe { out.store_unchecked(0, walk(values)) };
    unsafe { out.store_unchecked(1, walk(&copy)) };
    unsafe { out.store_unchecked(2, last) };
    unsafe { out.store_unchecked(3, offset as f32) };
}
