#![no_std]

pub use metal_oxide_macros::kernel;

pub mod intrinsics {
    unsafe extern "Rust" {
        #[cfg(not(bridge_flag))]
        pub fn __metal_matrix_load(
            buffer: *const (),
            row: u32,
            column: u32,
            stride: u32,
            transposed: bool,
            out: *mut (),
        );
        #[cfg(bridge_flag)]
        pub fn __metal_matrix_load(
            buffer: *const (),
            row: u32,
            column: u32,
            stride: u32,
            transposed: u32,
            out: *mut (),
        );
        pub fn __metal_matrix_store(
            buffer: *mut (),
            row: u32,
            column: u32,
            stride: u32,
            transposed: bool,
            matrix: *const (),
        );
    }
}

pub mod simdgroup {
    #[derive(Clone, Copy)]
    pub struct Matrix<T> {
        pub lanes: [T; 2],
    }
}

pub mod buffer {
    use crate::simdgroup::Matrix;

    #[derive(Clone, Copy)]
    pub struct ReadBuffer<T> {
        ptr: *const T,
    }

    #[cfg(wrong_buffer)]
    #[derive(Clone, Copy)]
    pub struct WriteBuffer<T> {
        ptr: *mut T,
    }

    impl<T: Copy> ReadBuffer<T> {
        #[cfg(safe_load)]
        pub fn load_matrix(self, row: u32, column: u32, stride: u32) -> Matrix<T> {
            let mut out = core::mem::MaybeUninit::<Matrix<T>>::uninit();
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast(),
                    row,
                    column,
                    stride,
                    false,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }

        #[cfg(other_element)]
        pub unsafe fn load_matrix(self, row: u32, column: u32, stride: u32) -> Matrix<f32> {
            let mut out = core::mem::MaybeUninit::<Matrix<f32>>::uninit();
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast(),
                    row,
                    column,
                    stride,
                    false,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }

        #[cfg(wrong_buffer)]
        pub unsafe fn load_matrix(
            buffer: WriteBuffer<T>,
            row: u32,
            column: u32,
            stride: u32,
        ) -> Matrix<T> {
            let mut out = core::mem::MaybeUninit::<Matrix<T>>::uninit();
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    buffer.ptr.cast_const().cast(),
                    row,
                    column,
                    stride,
                    false,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }

        #[cfg(bridge_flag)]
        pub unsafe fn load_matrix(self, row: u32, column: u32, stride: u32) -> Matrix<T> {
            let mut out = core::mem::MaybeUninit::<Matrix<T>>::uninit();
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast(),
                    row,
                    column,
                    stride,
                    0,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }

        #[cfg(two_calls)]
        pub unsafe fn load_matrix(self, row: u32, column: u32, stride: u32) -> Matrix<T> {
            let mut out = core::mem::MaybeUninit::<Matrix<T>>::uninit();
            unsafe {
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast(),
                    row,
                    column,
                    stride,
                    false,
                    out.as_mut_ptr().cast(),
                );
                crate::intrinsics::__metal_matrix_load(
                    self.ptr.cast(),
                    row,
                    column,
                    stride,
                    false,
                    out.as_mut_ptr().cast(),
                );
                out.assume_init()
            }
        }

        #[cfg(read_store)]
        pub unsafe fn store_matrix(self, row: u32, column: u32, stride: u32, matrix: Matrix<T>) {
            unsafe {
                crate::intrinsics::__metal_matrix_store(
                    self.ptr.cast_mut().cast(),
                    row,
                    column,
                    stride,
                    false,
                    (&raw const matrix).cast(),
                );
            }
        }
    }
}
