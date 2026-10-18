use core::mem::MaybeUninit;

use crate::{
    DynamicThreadgroupBuffer, ReadBuffer, ReadWriteBuffer, ThreadgroupBuffer, WriteBuffer,
    simdgroup::{Matrix, MatrixElement},
};

macro_rules! loads {
    () => {
        loads!(
            load_matrix,
            false,
            "The block's elements are initialized elements of this buffer that no thread \
             writes concurrently."
        );
        loads!(
            load_matrix_transposed,
            true,
            "As for [`load_matrix`](Self::load_matrix); the result is the block transposed."
        );
    };
    ($name:ident, $transposed:literal, $safety:literal) => {
        /// # Safety
        #[doc = $safety]
        pub unsafe fn $name(self, row: u32, column: u32, stride: u32) -> Matrix<T> {
            let mut out = MaybeUninit::<Matrix<T>>::uninit();
            // SAFETY: the caller guarantees bounds, initialization and no concurrent writes.
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast::<()>(),
                    row,
                    column,
                    stride,
                    $transposed,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }
    };
}

macro_rules! stores {
    () => {
        stores!(
            store_matrix,
            false,
            "The block lies in this buffer, `stride` is at least 8, and no other thread \
             accesses it concurrently."
        );
        stores!(
            store_matrix_transposed,
            true,
            "As for [`store_matrix`](Self::store_matrix); the block receives `matrix` transposed."
        );
    };
    ($name:ident, $transposed:literal, $safety:literal) => {
        /// # Safety
        #[doc = $safety]
        pub unsafe fn $name(self, row: u32, column: u32, stride: u32, matrix: Matrix<T>) {
            // SAFETY: the caller guarantees bounds, a stride of at least 8 and exclusive access.
            unsafe {
                crate::intrinsics::__metal_matrix_store(
                    self.ptr.cast::<()>(),
                    row,
                    column,
                    stride,
                    $transposed,
                    (&raw const matrix).cast(),
                );
            }
        }
    };
}

impl<T: MatrixElement> ReadBuffer<T> {
    loads!();
}

impl<T: MatrixElement> WriteBuffer<T> {
    stores!();
}

impl<T: MatrixElement> ReadWriteBuffer<T> {
    loads!();
    stores!();
}

impl<T: MatrixElement, const N: usize> ThreadgroupBuffer<T, N> {
    loads!();
    stores!();
}

impl<T: MatrixElement> DynamicThreadgroupBuffer<T> {
    loads!();
    stores!();
}
