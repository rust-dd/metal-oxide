#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, WriteBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn byte_triples(input: ReadBuffer<[u8; 3]>, mut output: DisjointBuffer<[u32; 4]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let position = index.get();
        let pixel = input.as_slice()[position];
        let channel = pixel[position % 3];
        let dark = (0_u32..3).any(move |c| pixel[(c + u32::from(channel)) as usize % 3] < 16);
        let wide = [u16::from(pixel[0]), u16::from(pixel[1]) * 3, u16::from(pixel[2]) * 5];
        let picked = (0_usize..2).fold(0_u16, move |total, k| total ^ wide[(k + position) % 3]);
        let bytes = [pixel[2], pixel[1], pixel[0], channel, 7];
        *output = [
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            u32::from(dark),
            u32::from(picked),
            u32::from(bytes[position % 5]),
        ];
    }
}

#[kernel]
pub unsafe fn reversed_triples(input: ReadBuffer<[u8; 3]>, output: WriteBuffer<[u8; 3]>, n: u32) {
    let index = thread_index().get() as u32;
    if index < n {
        let pixel = unsafe { input.load_unchecked(index) };
        unsafe { output.store_unchecked(index, [pixel[2], pixel[1], pixel[0]]) };
    }
}
