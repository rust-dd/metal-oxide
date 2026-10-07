#![no_std]

use metal_oxide_device::{WriteBuffer, kernel};

fn h0(x: u32) -> u32 {
    x
}
fn h1(x: u32) -> u32 {
    h0(x) + h0(x)
}
fn h2(x: u32) -> u32 {
    h1(x) + h1(x)
}
fn h3(x: u32) -> u32 {
    h2(x) + h2(x)
}
fn h4(x: u32) -> u32 {
    h3(x) + h3(x)
}
fn h5(x: u32) -> u32 {
    h4(x) + h4(x)
}
fn h6(x: u32) -> u32 {
    h5(x) + h5(x)
}
fn h7(x: u32) -> u32 {
    h6(x) + h6(x)
}
fn h8(x: u32) -> u32 {
    h7(x) + h7(x)
}
fn h9(x: u32) -> u32 {
    h8(x) + h8(x)
}
fn h10(x: u32) -> u32 {
    h9(x) + h9(x)
}
fn h11(x: u32) -> u32 {
    h10(x) + h10(x)
}
fn h12(x: u32) -> u32 {
    h11(x) + h11(x)
}
fn h13(x: u32) -> u32 {
    h12(x) + h12(x)
}
fn h14(x: u32) -> u32 {
    h13(x) + h13(x)
}
fn h15(x: u32) -> u32 {
    h14(x) + h14(x)
}
fn h16(x: u32) -> u32 {
    h15(x) + h15(x)
}

#[cfg(not(different_inputs))]
#[kernel]
pub unsafe fn proof_dag(out: WriteBuffer<u32>, value: u32) {
    if value < 2 {
        unsafe { out.store_unchecked(0, h16(value)) }
    }
}

#[cfg(different_inputs)]
fn divide(divisor: u32) -> u32 {
    64 / divisor
}

#[cfg(different_inputs)]
#[kernel]
pub unsafe fn proof_inputs(out: WriteBuffer<u32>, divisor: u32) {
    let known = divide(2);
    let unknown = divide(divisor);
    unsafe { out.store_unchecked(0, known + unknown) }
}
