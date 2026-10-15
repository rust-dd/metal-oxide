#![no_std]
use metal_oxide_device::{WriteBuffer, kernel};

struct Table<T>(T);

impl<T> Table<T> {
    const VALUES: [u32; 2] = {
        assert!(size_of::<T>() != 4);
        [1, 2]
    };
}

fn values<T>() -> [u32; 2] {
    Table::<T>::VALUES
}

#[kernel]
pub unsafe fn invalid(output: WriteBuffer<u32>) {
    unsafe { output.store_unchecked(0, values::<u32>()[0]) };
}
