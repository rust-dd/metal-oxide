#![no_std]
pub mod first {
    use metal_oxide_device::kernel;
    #[kernel]
    pub unsafe fn same() {}
}
pub mod second {
    use metal_oxide_device::kernel;
    #[kernel]
    pub unsafe fn same() {}
}
