#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

fn halve(value: u32) -> Result<u32, u32> {
    if value % 2 == 0 { Ok(value / 2) } else { Err(value) }
}

fn quarter(value: u32) -> Result<u32, u32> {
    let half = halve(value)?;
    halve(half)
}

/// # Safety
/// Nothing writes `values` while the kernel runs.
#[kernel]
pub unsafe fn quarters(values: ReadBuffer<u32>, mut out: DisjointBuffer<u32>) {
    let index = thread_index();
    let values = values.as_slice();
    let i = index.get();
    if let Some(out) = out.get_mut(&index)
        && i < values.len()
    {
        let first = values.get(0).copied().unwrap();
        let value = values.get(i).copied().expect("an index below the length");
        *out = match quarter(value) {
            Ok(quarter) => quarter + first,
            Err(odd) => odd,
        };
    }
}

/// # Safety
/// Nothing writes `values` while the kernel runs.
#[kernel]
pub unsafe fn past_the_end(values: ReadBuffer<u32>, mut out: DisjointBuffer<u32>) {
    let index = thread_index();
    let values = values.as_slice();
    if let Some(out) = out.get_mut(&index) {
        *out = values.get(values.len()).copied().unwrap();
    }
}
