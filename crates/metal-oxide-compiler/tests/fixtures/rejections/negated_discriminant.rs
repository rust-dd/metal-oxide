#![no_std]
#![feature(core_intrinsics)]
use metal_oxide_device::kernel;

enum Mode {
    Idle,
    Busy,
}

#[kernel]
pub unsafe fn invalid(flag: u32) {
    let mode = if flag == 0 { Mode::Idle } else { Mode::Busy };
    let _negated = -core::intrinsics::discriminant_value(&mode);
}
