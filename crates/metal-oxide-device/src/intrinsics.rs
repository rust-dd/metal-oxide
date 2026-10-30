use crate::{BF16, Dim3, F16};
use core::sync::atomic::Ordering;

/// Declares the device intrinsics that the compiler lowers; host builds, which type-check
/// kernels in ordinary crates, get definitions that panic.
macro_rules! intrinsics {
    ($($safety:ident fn $name:ident($($argument:ident: $ty:ty),* $(,)?) $(-> $output:ty)?;)*) => {
        #[cfg(target_env = "metal")]
        unsafe extern "Rust" {
            $(pub(crate) $safety fn $name($($argument: $ty),*) $(-> $output)?;)*
        }
        $(intrinsics!(@host $safety fn $name($($argument: $ty),*) $(-> $output)?);)*
    };
    (@host safe fn $name:ident($($argument:ident: $ty:ty),*) $(-> $output:ty)?) => {
        #[cfg(not(target_env = "metal"))]
        pub(crate) fn $name($(_: $ty),*) $(-> $output)? {
            host()
        }
    };
    (@host unsafe fn $name:ident($($argument:ident: $ty:ty),*) $(-> $output:ty)?) => {
        #[cfg(not(target_env = "metal"))]
        pub(crate) unsafe fn $name($(_: $ty),*) $(-> $output)? {
            host()
        }
    };
}

#[cfg(not(target_env = "metal"))]
#[cold]
fn host() -> ! {
    panic!("metal-oxide device functions run only in kernels on the GPU")
}

intrinsics! {
    unsafe fn __metal_atomic_load(ptr: *mut (), index: u32, out: *mut (), size: usize, order: Ordering);
    unsafe fn __metal_atomic_store(ptr: *mut (), index: u32, value: *const (), size: usize, order: Ordering);
    unsafe fn __metal_atomic_min_store(ptr: *mut (), index: u32, value: u64, order: Ordering);
    unsafe fn __metal_atomic_max_store(ptr: *mut (), index: u32, value: u64, order: Ordering);
    unsafe fn __metal_atomic_compare_exchange_weak(
        ptr: *mut (),
        index: u32,
        current: *const (),
        new: *const (),
        out: *mut (),
        size: usize,
        ordering: (Ordering, Ordering),
    ) -> bool;
    safe fn __metal_thread_idx() -> Dim3;
    safe fn __metal_thread_index() -> usize;
    safe fn __metal_block_idx() -> Dim3;
    safe fn __metal_block_dim() -> Dim3;
    safe fn __metal_grid_dim() -> Dim3;
    safe fn __metal_sqrt_f32(value: f32) -> f32;
    safe fn __metal_fma_f32(a: f32, b: f32, c: f32) -> f32;
    safe fn __metal_ln_f32(value: f32) -> f32;
    safe fn __metal_exp_f32(value: f32) -> f32;
    safe fn __metal_sin_f32(value: f32) -> f32;
    safe fn __metal_cos_f32(value: f32) -> f32;
    safe fn __metal_sin_cos_f32(value: f32) -> (f32, f32);
    safe fn __metal_powf_f32(base: f32, exponent: f32) -> f32;
    safe fn __metal_tanh_f32(value: f32) -> f32;
    safe fn __metal_atan_f32(value: f32) -> f32;
    safe fn __metal_floor_f32(value: f32) -> f32;
    safe fn __metal_round_f32(value: f32) -> f32;
    safe fn __metal_ceil_f32(value: f32) -> f32;
    safe fn __metal_trunc_f32(value: f32) -> f32;
    safe fn __metal_exp2_f32(value: f32) -> f32;
    safe fn __metal_log2_f32(value: f32) -> f32;
    safe fn __metal_log10_f32(value: f32) -> f32;
    safe fn __metal_tan_f32(value: f32) -> f32;
    safe fn __metal_asin_f32(value: f32) -> f32;
    safe fn __metal_acos_f32(value: f32) -> f32;
    safe fn __metal_atan2_f32(y: f32, x: f32) -> f32;
    safe fn __metal_rsqrt_f32(value: f32) -> f32;
    safe fn __metal_f16_from_f32(value: f32) -> F16;
    safe fn __metal_f16_to_f32(value: F16) -> f32;
    safe fn __metal_f16_add(a: F16, b: F16) -> F16;
    safe fn __metal_f16_sub(a: F16, b: F16) -> F16;
    safe fn __metal_f16_mul(a: F16, b: F16) -> F16;
    safe fn __metal_f16_div(a: F16, b: F16) -> F16;
    safe fn __metal_f16_rem(a: F16, b: F16) -> F16;
    safe fn __metal_bf16_add(a: BF16, b: BF16) -> BF16;
    safe fn __metal_bf16_sub(a: BF16, b: BF16) -> BF16;
    safe fn __metal_bf16_mul(a: BF16, b: BF16) -> BF16;
    safe fn __metal_bf16_div(a: BF16, b: BF16) -> BF16;
    safe fn __metal_bf16_rem(a: BF16, b: BF16) -> BF16;
    safe fn __metal_bf16_from_f32(value: f32) -> BF16;
    safe fn __metal_bf16_to_f32(value: BF16) -> f32;
    safe fn __metal_simd_lane() -> u32;
    safe fn __metal_simd_size() -> u32;
    safe fn __metal_simd_group() -> u32;
    safe fn __metal_simd_count() -> u32;
    unsafe fn __metal_threadgroup_barrier();
    unsafe fn __metal_log(format: *const (), length: usize, arguments: *const (), bytes: usize);
    unsafe fn __metal_uninit(out: *mut (), bytes: usize);
    unsafe fn __metal_threadgroup_alloc(bytes: usize, alignment: usize) -> *mut ();
    unsafe fn __metal_buffer_length(buffer: *const ()) -> usize;
    unsafe fn __metal_buffer_load(buffer: *const (), index: u32, out: *mut (), bytes: usize);
    unsafe fn __metal_buffer_store(buffer: *mut (), index: u32, value: *const (), bytes: usize);
    unsafe fn __metal_threadgroup_load(buffer: *const (), index: u32, out: *mut (), bytes: usize);
    unsafe fn __metal_threadgroup_store(
        buffer: *mut (),
        index: u32,
        value: *const (),
        bytes: usize,
    );
    unsafe fn __metal_simd_any(predicate: bool) -> bool;
    unsafe fn __metal_simd_all(predicate: bool) -> bool;
    unsafe fn __metal_simd_ballot(predicate: bool) -> [u32; 2];
    unsafe fn __metal_matrix_splat(value: *const (), out: *mut (), size: usize);
    unsafe fn __metal_matrix_multiply_accumulate(
        a: *const (),
        b: *const (),
        c: *const (),
        out: *mut (),
    );
    unsafe fn __metal_matrix_load(
        buffer: *const (),
        row: u32,
        column: u32,
        stride: u32,
        transposed: bool,
        out: *mut (),
    );
    unsafe fn __metal_matrix_store(
        buffer: *mut (),
        row: u32,
        column: u32,
        stride: u32,
        transposed: bool,
        matrix: *const (),
    );
}

macro_rules! atomic_modify {
    ($($name:ident),* $(,)?) => {
        intrinsics! {
            $(unsafe fn $name(ptr: *mut (), index: u32, value: *const (), out: *mut (), size: usize, order: Ordering);)*
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
        intrinsics! {
            $(unsafe fn $name(value: *const (), out: *mut (), size: usize);)*
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
        intrinsics! {
            $(unsafe fn $name(value: *const (), control: u32, out: *mut (), size: usize);)*
        }
    };
}
simd_permute!(
    __metal_simd_shuffle,
    __metal_simd_shuffle_up,
    __metal_simd_shuffle_down,
    __metal_simd_shuffle_xor
);
