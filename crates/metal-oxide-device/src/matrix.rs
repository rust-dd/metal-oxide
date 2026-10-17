use core::mem::MaybeUninit;

use crate::{
    DynamicThreadgroupBuffer, ReadBuffer, ReadWriteBuffer, ThreadgroupBuffer, WriteBuffer,
    simdgroup::{Matrix, MatrixElement},
};

macro_rules! loads {
    () => {
        /// # Safety
        /// The block's elements are initialized elements of this buffer that no thread writes concurrently.
        pub unsafe fn load_matrix(self, row: u32, column: u32, stride: u32) -> Matrix<T> {
            let mut out = MaybeUninit::<Matrix<T>>::uninit();
            // SAFETY: the caller guarantees the block's bounds, initialization and access discipline.
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast::<()>(),
                    row,
                    column,
                    stride,
                    false,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }

        /// # Safety
        /// As for [`load_matrix`](Self::load_matrix); the result is the block transposed.
        pub unsafe fn load_matrix_transposed(
            self,
            row: u32,
            column: u32,
            stride: u32,
        ) -> Matrix<T> {
            let mut out = MaybeUninit::<Matrix<T>>::uninit();
            // SAFETY: the caller guarantees the block's bounds, initialization and access discipline.
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast::<()>(),
                    row,
                    column,
                    stride,
                    true,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }
    };
}

macro_rules! stores {
    () => {
        /// # Safety
        /// The block lies in this buffer, `stride` is at least 8, and no other thread accesses it concurrently.
        pub unsafe fn store_matrix(self, row: u32, column: u32, stride: u32, matrix: Matrix<T>) {
            // SAFETY: the caller guarantees the block's bounds and exclusive access.
            unsafe {
                crate::intrinsics::__metal_matrix_store(
                    self.ptr.cast::<()>(),
                    row,
                    column,
                    stride,
                    false,
                    (&raw const matrix).cast(),
                );
            }
        }

        /// # Safety
        /// As for [`store_matrix`](Self::store_matrix); the block receives `matrix` transposed.
        pub unsafe fn store_matrix_transposed(
            self,
            row: u32,
            column: u32,
            stride: u32,
            matrix: Matrix<T>,
        ) {
            // SAFETY: the caller guarantees the block's bounds and exclusive access.
            unsafe {
                crate::intrinsics::__metal_matrix_store(
                    self.ptr.cast::<()>(),
                    row,
                    column,
                    stride,
                    true,
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
