#![no_std]

pub use metal_oxide_macros::kernel;

pub mod intrinsics {
    unsafe extern "Rust" {
        pub fn __metal_threadgroup_alloc(bytes: usize, alignment: usize) -> *mut ();
    }
}

pub mod threadgroup {
    #[derive(Clone, Copy)]
    pub struct ThreadgroupBuffer<T, const N: usize> {
        ptr: *mut T,
    }

    #[cfg(reversed)]
    pub fn shared<const N: usize, T: Copy>() -> ThreadgroupBuffer<T, N> {
        let ptr = unsafe {
            crate::intrinsics::__metal_threadgroup_alloc(
                core::mem::size_of::<T>() * N,
                core::mem::align_of::<T>(),
            )
        };
        ThreadgroupBuffer { ptr: ptr.cast() }
    }

    #[cfg(fixed_length)]
    pub fn shared<T: Copy, const N: usize>() -> ThreadgroupBuffer<T, 1> {
        let ptr = unsafe {
            crate::intrinsics::__metal_threadgroup_alloc(
                core::mem::size_of::<T>(),
                core::mem::align_of::<T>(),
            )
        };
        ThreadgroupBuffer { ptr: ptr.cast() }
    }

    #[cfg(fixed_element)]
    pub fn shared<T: Copy, const N: usize>() -> ThreadgroupBuffer<u32, N> {
        let ptr = unsafe {
            crate::intrinsics::__metal_threadgroup_alloc(
                core::mem::size_of::<u32>() * N,
                core::mem::align_of::<u32>(),
            )
        };
        ThreadgroupBuffer { ptr: ptr.cast() }
    }
}
