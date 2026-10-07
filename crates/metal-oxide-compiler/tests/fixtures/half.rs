#![no_std]

use metal_oxide_device::{F16, ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx};

#[derive(Clone, Copy)]
pub struct HalfRecord {
    pub tag: u8,
    pub pair: (F16, F16),
    pub samples: [F16; 2],
}

#[kernel]
pub unsafe fn half_bits(input: ReadBuffer<F16>, bits: WriteBuffer<u16>, wide: WriteBuffer<f32>, copied: WriteBuffer<F16>, n: u32) {
    let i = block_idx().x.wrapping_mul(block_dim().x).wrapping_add(thread_idx().x);
    if i < n {
        unsafe {
            let value = input.load_unchecked(i);
            bits.store_unchecked(i, value.to_bits());
            wide.store_unchecked(i, value.to_f32());
            copied.store_unchecked(i, F16::from_bits(value.to_bits()));
        }
    }
}

#[kernel]
pub unsafe fn half_round(input: ReadBuffer<f32>, out: WriteBuffer<F16>, n: u32) {
    let i = block_idx().x.wrapping_mul(block_dim().x).wrapping_add(thread_idx().x);
    if i < n { unsafe { out.store_unchecked(i, F16::from_f32(input.load_unchecked(i))); } }
}

#[kernel]
pub unsafe fn half_record(out: WriteBuffer<f32>, value: HalfRecord, scale: F16) {
    const VALUES: [F16; 2] = [F16::from_bits(0x3c00), F16::from_bits(0x8001)];
    let copied = value;
    unsafe {
        out.store_unchecked(0, copied.pair.0.to_f32() * scale.to_f32());
        out.store_unchecked(1, copied.pair.1.to_f32());
        out.store_unchecked(2, copied.samples[0].to_f32());
        out.store_unchecked(3, copied.samples[1].to_f32());
        out.store_unchecked(4, VALUES[0].to_f32());
        out.store_unchecked(5, VALUES[1].to_f32());
    }
}
