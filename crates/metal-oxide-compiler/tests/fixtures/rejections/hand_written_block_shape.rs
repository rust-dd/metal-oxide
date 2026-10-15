#![no_std]
use metal_oxide_device::kernel;
// `#[kernel(block = ...)]` rejects a zero axis; this attribute bypasses that check.
#[kernel]
#[metal_oxide_compiler::block_shape = "0,1,1"]
pub unsafe fn hand_written_block_shape() {}
