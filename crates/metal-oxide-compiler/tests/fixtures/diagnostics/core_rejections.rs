#![no_std]
use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel};

#[cfg(import)]
#[kernel]
pub unsafe fn reached(input: ReadBuffer<u32>, out: WriteBuffer<u32>) {
    let values = unsafe { [input.load_unchecked(0), input.load_unchecked(1)] };
    unsafe { out.store_unchecked(0, values.into_iter().sum::<u32>()) };
}

#[cfg(collector)]
#[kernel]
pub unsafe fn reached(input: ReadBuffer<u32>, out: WriteBuffer<u32>) {
    let mut values = unsafe { [input.load_unchecked(0), input.load_unchecked(1)] };
    values.rotate_left(1);
    unsafe { out.store_unchecked(0, values[0]) };
}

#[cfg(proof)]
#[kernel]
pub unsafe fn reached(input: ReadBuffer<u32>, out: WriteBuffer<u32>) {
    let values = unsafe { [input.load_unchecked(0), input.load_unchecked(1)] };
    let index = unsafe { input.load_unchecked(2) } as usize;
    unsafe { out.store_unchecked(0, *values.get_unchecked(index)) };
}
