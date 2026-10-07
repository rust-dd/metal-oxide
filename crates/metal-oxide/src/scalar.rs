mod sealed {
    pub trait Atomic {}
    impl Atomic for u32 {}
    impl Atomic for i32 {}
}

/// Integer elements supported by device atomic operations.
pub trait GpuAtomic: crate::GpuValue + sealed::Atomic {}
impl GpuAtomic for u32 {}
impl GpuAtomic for i32 {}
