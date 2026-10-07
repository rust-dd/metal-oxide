#![no_std]
#![allow(non_camel_case_types)]

use metal_oxide_device::kernel;

#[derive(Clone, Copy)]
pub struct usize {
    pub value: u32,
}

#[derive(Clone, Copy)]
pub struct AsRef {
    pub value: u32,
}

#[derive(Clone, Copy)]
pub struct Ok {
    pub value: u32,
}

#[derive(Clone, Copy)]
pub struct Into {
    pub value: u32,
}

#[kernel]
pub unsafe fn named(_size: usize, _reference: AsRef, _result: Ok, _conversion: Into) {}
