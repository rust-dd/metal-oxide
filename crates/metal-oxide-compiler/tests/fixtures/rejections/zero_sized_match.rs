#![no_std]
use core::convert::Infallible;
use metal_oxide_device::{WriteBuffer, kernel};

enum Choice {
    Taken,
    Never(Infallible),
}

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    // `Never` is uninhabited, so `Choice` and the tuple around it are zero-sized.
    let wrapped = (Choice::Taken,);
    let value = match wrapped.0 {
        Choice::Never(_) => 2,
        Choice::Taken => 1,
    };
    unsafe { output.store_unchecked(0, value) };
}
