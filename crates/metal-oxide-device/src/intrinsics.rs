use crate::{Dim3, F16};

unsafe extern "Rust" {
    pub(crate) fn __metal_atomic_load(ptr: *mut (), index: u32, out: *mut (), size: usize);
    pub(crate) fn __metal_atomic_store(ptr: *mut (), index: u32, value: *const (), size: usize);
    pub(crate) fn __metal_atomic_compare_exchange_weak(
        ptr: *mut (),
        index: u32,
        current: *const (),
        new: *const (),
        out: *mut (),
        size: usize,
    ) -> bool;
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
}

macro_rules! atomic_modify {
    ($($name:ident),* $(,)?) => {
        unsafe extern "Rust" {
            $(pub(crate) fn $name(ptr: *mut (), index: u32, value: *const (), out: *mut (), size: usize);)*
        }
    };
}
atomic_modify!(
    __metal_atomic_exchange,
    __metal_atomic_add,
    __metal_atomic_sub,
    __metal_atomic_min,
    __metal_atomic_max,
    __metal_atomic_and,
    __metal_atomic_or,
    __metal_atomic_xor
);

macro_rules! simd_reduce {
    ($($name:ident),* $(,)?) => {
        unsafe extern "Rust" {
            $(pub(crate) fn $name(value: *const (), out: *mut (), size: usize);)*
        }
    };
}
simd_reduce!(
    __metal_simd_sum,
    __metal_simd_min,
    __metal_simd_max,
    __metal_simd_and,
    __metal_simd_or,
    __metal_simd_xor,
    __metal_simd_inclusive_sum,
    __metal_simd_exclusive_sum
);
macro_rules! simd_permute {
    ($($name:ident),* $(,)?) => {
        unsafe extern "Rust" {
            $(pub(crate) fn $name(value: *const (), control: u32, out: *mut (), size: usize);)*
        }
    };
}
simd_permute!(
    __metal_simd_shuffle,
    __metal_simd_shuffle_up,
    __metal_simd_shuffle_down,
    __metal_simd_shuffle_xor
);
unsafe extern "Rust" {
    pub(crate) fn __metal_simd_any(predicate: bool) -> bool;
    pub(crate) fn __metal_simd_all(predicate: bool) -> bool;
    pub(crate) fn __metal_simd_ballot(predicate: bool) -> [u32; 2];
}
