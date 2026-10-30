#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn byte_conversions(
    input: ReadBuffer<(u64, i32, i16)>,
    mut output: DisjointBuffer<([u8; 8], [u8; 4], [u8; 2], [u16; 2], [u64; 5])>,
) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let (wide, signed, short) = input.as_slice()[index.get()];
        let little = wide.to_le_bytes();
        let big = signed.to_be_bytes();
        let native = short.to_ne_bytes();
        let halves = unsafe { core::mem::transmute::<u32, [u16; 2]>(signed as u32) };
        let lanes = [short, short.wrapping_neg(), short ^ 7, 1];
        let packed = unsafe { core::mem::transmute::<[i16; 4], u64>(lanes) };
        *output = (
            little,
            big,
            native,
            halves,
            [
                u64::from_be_bytes(little),
                i32::from_le_bytes(big) as u64,
                i16::from_be_bytes(native) as u64,
                f32::from_le_bytes((wide as f32).to_be_bytes()).to_bits() as u64,
                packed,
            ],
        );
    }
}
