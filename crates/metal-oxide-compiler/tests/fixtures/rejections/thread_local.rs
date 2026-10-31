#![no_std]
#![feature(thread_local)]
use metal_oxide_device::{WriteBuffer, kernel};

#[thread_local]
static COUNTER: u32 = 7;

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    unsafe { output.store_unchecked(0, COUNTER) };
}
