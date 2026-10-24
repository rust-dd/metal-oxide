#![no_std]

use metal_oxide_device::{ReadBuffer, WriteBuffer, kernel, thread_idx};

#[cfg(call_mutation)]
fn replace(index: &mut u32, other: u32) {
    *index = other;
}

#[kernel]
pub unsafe fn bad(input: ReadBuffer<u32>, output: WriteBuffer<u32>, other: u32) {
    let values = input.as_slice();
    #[cfg(narrowing)]
    let index = other;
    #[cfg(not(narrowing))]
    let mut index = thread_idx().x;
    #[cfg(narrowing)]
    let checked = index as u8 as usize;
    #[cfg(not(narrowing))]
    let checked = index as usize;
    #[cfg(stale_guard)]
    {
        index = other;
    }
    if checked < values.len() {
        #[cfg(reassign)]
        {
            index = other;
        }
        #[cfg(call_mutation)]
        replace(&mut index, other);
        #[cfg(join)]
        if index != 0 {
            index = other;
        }
        #[cfg(loop_mutation)]
        for _ in 0..2 {
            unsafe { output.store_unchecked(thread_idx().x, values[index as usize]) };
            index = other;
        }
        #[cfg(not(loop_mutation))]
        unsafe {
            output.store_unchecked(thread_idx().x, values[index as usize])
        };
    }
    #[cfg(barrier)]
    unsafe {
        metal_oxide_device::threadgroup::barrier()
    };
}
