//! Collectives over active SIMD lanes. Numeric operations and shuffles support f32/u32/i32;
//! bit reductions support u32/i32. Integer sums wrap; float reduction order is Metal's.
//! The compiler currently requires uniform block participation for collectives.

use core::mem::{MaybeUninit, size_of};

/// The current lane within its SIMD group.
pub fn lane_id() -> u32 {
    crate::intrinsics::__metal_simd_lane()
}

/// The number of lanes in a SIMD group.
pub fn size() -> u32 {
    crate::intrinsics::__metal_simd_size()
}

/// The current SIMD group within its block.
pub fn group_id() -> u32 {
    crate::intrinsics::__metal_simd_group()
}

/// The number of SIMD groups in the current block.
pub fn groups_per_block() -> u32 {
    crate::intrinsics::__metal_simd_count()
}

macro_rules! reduce {
    ($name:ident, $bridge:ident, $description:literal) => {
        #[doc = $description]
        ///
        /// # Safety
        /// Every active lane must reach this operation in the same iteration.
        pub unsafe fn $name<T: Copy>(value: T) -> T {
            let mut out = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies uniform participation; the bridge initializes T.
            unsafe {
                crate::intrinsics::$bridge(
                    (&raw const value).cast(),
                    out.as_mut_ptr().cast(),
                    size_of::<T>(),
                );
                out.assume_init()
            }
        }
    };
}
reduce!(
    sum,
    __metal_simd_sum,
    "Sums active lanes and broadcasts the result."
);
reduce!(
    min,
    __metal_simd_min,
    "Takes the minimum across active lanes."
);
reduce!(
    max,
    __metal_simd_max,
    "Takes the maximum across active lanes."
);
reduce!(
    and,
    __metal_simd_and,
    "Computes bitwise AND across active lanes."
);
reduce!(
    or,
    __metal_simd_or,
    "Computes bitwise OR across active lanes."
);
reduce!(
    xor,
    __metal_simd_xor,
    "Computes bitwise XOR across active lanes."
);
reduce!(
    inclusive_sum,
    __metal_simd_inclusive_sum,
    "Sums active lanes up to and including the current lane."
);
reduce!(
    exclusive_sum,
    __metal_simd_exclusive_sum,
    "Sums preceding active lanes; the first lane receives zero."
);

macro_rules! permute {
    ($name:ident, $bridge:ident, $description:literal, $contract:literal) => {
        #[doc = $description]
        ///
        /// # Safety
        /// Every active lane must reach this operation in the same iteration.
        #[doc = $contract]
        pub unsafe fn $name<T: Copy>(value: T, control: u32) -> T {
            let mut out = MaybeUninit::<T>::uninit();
            // SAFETY: the caller supplies participation and valid sources; the bridge initializes T.
            unsafe {
                crate::intrinsics::$bridge(
                    (&raw const value).cast(),
                    control,
                    out.as_mut_ptr().cast(),
                    size_of::<T>(),
                );
                out.assume_init()
            }
        }
    };
}
permute!(
    shuffle,
    __metal_simd_shuffle,
    "Reads another lane's value.",
    "The source lane must be active and below size(); each lane may choose its own source."
);
permute!(
    shuffle_up,
    __metal_simd_shuffle_up,
    "Reads lane - delta; lower delta lanes retain their own value.",
    "Delta must be uniform and below size(). Every in-range source lane must be active."
);
permute!(
    shuffle_down,
    __metal_simd_shuffle_down,
    "Reads lane + delta; upper delta lanes retain their own value.",
    "Delta must be uniform and below size(). Every in-range source lane must be active."
);
permute!(
    shuffle_xor,
    __metal_simd_shuffle_xor,
    "Reads lane XOR mask.",
    "Mask must be uniform and below size(). Every source lane must be active."
);

macro_rules! vote {
    ($name:ident, $bridge:ident, $result:ty, $description:literal) => {
        #[doc = $description]
        ///
        /// # Safety
        /// Every active lane must reach this operation in the same iteration.
        pub unsafe fn $name(predicate: bool) -> $result {
            // SAFETY: the caller guarantees uniform participation.
            unsafe { crate::intrinsics::$bridge(predicate) }
        }
    };
}
vote!(
    any,
    __metal_simd_any,
    bool,
    "Returns true if any active lane supplies true."
);
vote!(
    all,
    __metal_simd_all,
    bool,
    "Returns true if every active lane supplies true."
);
vote!(
    ballot,
    __metal_simd_ballot,
    [u32; 2],
    "Returns low/high 32-bit vote words. Bit i represents lane i; inactive and unused bits are zero."
);

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for crate::F16 {}
    impl Sealed for crate::BF16 {}
}

/// The element types of [`Matrix`]: f32, [`F16`](crate::F16) and [`BF16`](crate::BF16).
pub trait MatrixElement: Copy + sealed::Sealed {}
impl MatrixElement for f32 {}
impl MatrixElement for crate::F16 {}
impl MatrixElement for crate::BF16 {}

/// An 8x8 matrix that a SIMD group's 32 lanes hold. A load or store addresses element `(i, j)` of
/// its block at `(row + i) * stride + column + j`; all lanes pass the same operands.
#[derive(Clone, Copy)]
pub struct Matrix<T: MatrixElement> {
    #[expect(dead_code, reason = "the GPU keeps a matrix in lane registers")]
    lanes: [T; 2],
}

impl<T: MatrixElement> Matrix<T> {
    /// A matrix with every element set to `value`.
    pub fn splat(value: T) -> Self {
        let mut out = MaybeUninit::<Self>::uninit();
        // SAFETY: the bridge initializes the matrix.
        unsafe {
            crate::intrinsics::__metal_matrix_splat(
                (&raw const value).cast(),
                out.as_mut_ptr().cast(),
                size_of::<T>(),
            );
            out.assume_init()
        }
    }

    /// `a * b + c`, rounded as Metal's SIMD-group multiply-accumulate rounds.
    pub fn multiply_accumulate<A: MatrixElement, B: MatrixElement>(
        a: Matrix<A>,
        b: Matrix<B>,
        c: Self,
    ) -> Self {
        let mut out = MaybeUninit::<Self>::uninit();
        // SAFETY: the bridge reads three matrices and initializes the result.
        unsafe {
            crate::intrinsics::__metal_matrix_multiply_accumulate(
                (&raw const a).cast(),
                (&raw const b).cast(),
                (&raw const c).cast(),
                out.as_mut_ptr().cast(),
            );
            out.assume_init()
        }
    }
}
