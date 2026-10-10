#![no_std]

use metal_oxide_device::{DynamicThreadgroupBuffer, kernel, thread_idx, threadgroup::barrier};

unsafe fn helper(values: DynamicThreadgroupBuffer<u32>) {
    unsafe {
        values.store_unchecked(thread_idx().x, 1);
        barrier();
    }
}

#[kernel]
pub unsafe fn divergent(values: DynamicThreadgroupBuffer<u32>) {
    if thread_idx().x > 0 {
        unsafe { helper(values) };
    }
}
