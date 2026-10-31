#![no_std]

use metal_oxide_device::{DisjointBuffer, ReadBuffer, kernel, thread_index};

#[kernel(requires(input.len() == output.len()))]
pub fn iterated(input: ReadBuffer<[u32; 4]>, mut output: DisjointBuffer<[u32; 5]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let values = input.as_slice()[index.get()];
        let mut total = 0_u32;
        for value in values {
            total = total.wrapping_mul(31).wrapping_add(value);
        }
        let mut weighted = 0_u32;
        for (i, value) in values.into_iter().enumerate() {
            weighted = weighted.wrapping_add(value.wrapping_mul(i as u32 + 1));
        }
        let mut dot = 0_u32;
        for (x, y) in values.into_iter().zip([3_u32, 5, 7, 11]) {
            dot = dot.wrapping_add(x.wrapping_mul(y));
        }
        let mut nested = 0_u32;
        for row in [[values[0], values[1]], [values[2], values[3]]] {
            for value in row {
                nested = nested.wrapping_mul(7) ^ value;
            }
        }
        let mut iterator = values.into_iter();
        let head = iterator.next().unwrap_or(0) ^ iterator.next().unwrap_or(0);
        let mut rest = 0_u32;
        while let Some(value) = iterator.next() {
            rest = rest.wrapping_add(value);
        }
        *output = [total, weighted, dot, nested, head.wrapping_mul(rest)];
    }
}

#[kernel(requires(input.len() == output.len()))]
pub fn built(input: ReadBuffer<[u32; 4]>, mut output: DisjointBuffer<[u32; 6]>) {
    let index = thread_index();
    if let Some(output) = output.get_mut(&index) {
        let values = input.as_slice()[index.get()];
        let tripled = values.map(|value| value.wrapping_mul(3));
        let key = values[1];
        let keyed = values.map(|value| value ^ key);
        let mut count = 0_u32;
        let counted = values.map(|value| {
            count += 1;
            value.wrapping_add(count)
        });
        let halves = values.map(|value| value as u8 as f32 * 0.5);
        let mirrored = core::array::from_fn::<u32, 4, _>(|i| values[i].wrapping_add(values[3 - i]));
        let grid = core::array::from_fn::<[u32; 3], 2, _>(|i| {
            core::array::from_fn(|j| values[i + j].wrapping_mul(j as u32))
        });
        *output = [
            tripled[0] ^ tripled[3],
            keyed[2],
            counted[3].wrapping_add(count),
            halves[1].to_bits() ^ halves[2].to_bits(),
            mirrored[1] ^ mirrored[3],
            grid[1][2] ^ grid[0][1],
        ];
    }
}
