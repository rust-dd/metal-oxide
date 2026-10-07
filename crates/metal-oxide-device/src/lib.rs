#![no_std]

mod atomic;
mod buffer;
pub mod math;
pub mod simdgroup;
mod thread;
pub mod threadgroup;

pub use atomic::AtomicBuffer;
pub use buffer::{ReadBuffer, WriteBuffer};
pub use metal_oxide_macros::kernel;
pub use thread::{Dim3, block_dim, block_idx, grid_dim, thread_idx};
pub use threadgroup::ThreadgroupBuffer;
