#![no_std]

mod atomic;
mod buffer;
mod half;
mod intrinsics;
pub mod log;
pub mod math;
pub mod simdgroup;
mod thread;
pub mod threadgroup;

/// Kernels written in a host crate name this API `metal_oxide::device`, also in their device build.
pub use crate as device;
pub use atomic::{AtomicBuffer, AtomicThreadgroupBuffer};
pub use buffer::{DisjointBuffer, ReadBuffer, ReadWriteBuffer, WriteBuffer};
pub use half::F16;
pub use metal_oxide_macros::kernel;
pub use thread::{
    Dim3, ThreadIndex, block_dim, block_idx, grid_dim, thread_idx, thread_index, uninit_array,
};
pub use threadgroup::{DynamicThreadgroupBuffer, ThreadgroupBuffer};
