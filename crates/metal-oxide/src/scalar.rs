mod sealed {
    pub trait Atomic {}
    impl Atomic for u32 {}
    impl Atomic for i32 {}
    impl Atomic for f32 {}
    impl Atomic for u64 {}
}

/// Elements supported by device atomic operations, subject to each operation's capabilities.
pub trait GpuAtomic: crate::GpuValue + sealed::Atomic {}
impl GpuAtomic for u32 {}
impl GpuAtomic for i32 {}
impl GpuAtomic for f32 {}
impl GpuAtomic for u64 {}
