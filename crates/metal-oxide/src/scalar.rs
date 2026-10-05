mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
}

/// Buffer elements with a defined four-byte Metal representation.
///
/// This trait is sealed: arbitrary Rust layouts cannot cross the GPU boundary.
///
/// ```compile_fail
/// use metal_oxide::GpuScalar;
/// fn buffer_element<T: GpuScalar>() {}
/// buffer_element::<bool>();
/// ```
pub trait GpuScalar: sealed::Sealed + Copy + 'static {}

impl GpuScalar for f32 {}
impl GpuScalar for u32 {}
impl GpuScalar for i32 {}
