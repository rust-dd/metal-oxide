use crate::{Dim3, F16};

unsafe extern "Rust" {
    pub(crate) safe fn __metal_thread_idx() -> Dim3;
    pub(crate) safe fn __metal_block_idx() -> Dim3;
    pub(crate) safe fn __metal_block_dim() -> Dim3;
    pub(crate) safe fn __metal_grid_dim() -> Dim3;
    pub(crate) safe fn __metal_sqrt_f32(value: f32) -> f32;
    pub(crate) safe fn __metal_fma_f32(a: f32, b: f32, c: f32) -> f32;
    pub(crate) safe fn __metal_f16_from_f32(value: f32) -> F16;
    pub(crate) safe fn __metal_f16_to_f32(value: F16) -> f32;
    pub(crate) safe fn __metal_simd_lane() -> u32;
    pub(crate) safe fn __metal_simd_size() -> u32;
    pub(crate) safe fn __metal_simd_group() -> u32;
    pub(crate) safe fn __metal_simd_count() -> u32;
    pub(crate) fn __metal_simd_sum(value: f32) -> f32;
    pub(crate) fn __metal_simd_shuffle(value: f32, lane: u32) -> f32;
    pub(crate) fn __metal_threadgroup_barrier();
    pub(crate) fn __metal_threadgroup_alloc(bytes: usize, alignment: usize) -> *mut ();
    pub(crate) fn __metal_buffer_load(buffer: *const (), index: u32, out: *mut (), bytes: usize);
    pub(crate) fn __metal_buffer_store(buffer: *mut (), index: u32, value: *const (), bytes: usize);
    pub(crate) fn __metal_threadgroup_load(
        buffer: *const (),
        index: u32,
        out: *mut (),
        bytes: usize,
    );
    pub(crate) fn __metal_threadgroup_store(
        buffer: *mut (),
        index: u32,
        value: *const (),
        bytes: usize,
    );
    pub(crate) fn __metal_atomic_add(
        buffer: *mut (),
        index: u32,
        value: *const (),
        out: *mut (),
        bytes: usize,
    );
}
