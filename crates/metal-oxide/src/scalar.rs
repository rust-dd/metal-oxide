mod sealed {
    pub trait Atomic {}
    impl Atomic for u32 {}
    impl Atomic for i32 {}
    pub trait Sealed {
        const TYPE: metal_oxide_artifact::Scalar;
    }

    impl Sealed for f32 {
        const TYPE: metal_oxide_artifact::Scalar = metal_oxide_artifact::Scalar::F32;
    }
    impl Sealed for u32 {
        const TYPE: metal_oxide_artifact::Scalar = metal_oxide_artifact::Scalar::U32;
    }
    impl Sealed for i32 {
        const TYPE: metal_oxide_artifact::Scalar = metal_oxide_artifact::Scalar::I32;
    }
    impl Sealed for u8 {
        const TYPE: metal_oxide_artifact::Scalar = metal_oxide_artifact::Scalar::U8;
    }
    impl Sealed for u16 {
        const TYPE: metal_oxide_artifact::Scalar = metal_oxide_artifact::Scalar::U16;
    }
}

/// Integer buffer elements supported by device atomic operations.
pub trait GpuAtomic: GpuScalar + sealed::Atomic {}
impl GpuAtomic for u32 {}
impl GpuAtomic for i32 {}

/// Scalar buffer elements with a defined Metal representation.
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

impl GpuScalar for u8 {}
impl GpuScalar for u16 {}
