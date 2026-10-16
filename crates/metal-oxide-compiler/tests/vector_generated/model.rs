use crate::generated_support::Random;

pub const SCALARS: [&str; 7] = ["f32", "i32", "u32", "i8", "u8", "i16", "u16"];
pub const SEEDS: [u64; 3] = [0, 0x0123_4567_89ab_cdef, u64::MAX];

#[derive(Clone, Debug)]
pub enum Op {
    Array,
    Splat,
    ArrayReference,
    Insert,
    Swizzle(bool),
    Reverse,
    Rotate(bool),
    Binary(&'static str),
    Method(&'static str),
    Compare(&'static str),
    Mask(&'static str),
    Reduction(&'static str),
    Cast(&'static str),
}

pub fn operations(scalar: &str) -> Vec<Op> {
    let mut result = vec![
        Op::Array,
        Op::Splat,
        Op::ArrayReference,
        Op::Insert,
        Op::Swizzle(false),
        Op::Swizzle(true),
        Op::Reverse,
        Op::Rotate(false),
        Op::Rotate(true),
    ];
    result.extend(["+", "-", "*", "/", "%"].map(Op::Binary));
    result.extend(["simd_min", "simd_max", "simd_clamp"].map(Op::Method));
    result.extend(
        [
            "simd_eq", "simd_ne", "simd_lt", "simd_le", "simd_gt", "simd_ge",
        ]
        .map(Op::Compare),
    );
    result.extend(
        [
            "array", "splat", "bitmask", "and", "or", "xor", "not", "any", "all",
        ]
        .map(Op::Mask),
    );
    result.extend(["reduce_sum", "reduce_product", "reduce_min", "reduce_max"].map(Op::Reduction));
    result.extend(SCALARS.map(Op::Cast));
    if scalar != "f32" {
        result.extend(["&", "|", "^", "<<", ">>"].map(Op::Binary));
        result.extend(
            [
                "not",
                "saturating_add",
                "saturating_sub",
                "reverse_bits",
                "count_ones",
                "count_zeros",
                "leading_zeros",
                "trailing_zeros",
                "leading_ones",
                "trailing_ones",
            ]
            .map(Op::Method),
        );
        result.extend(["reduce_and", "reduce_or", "reduce_xor"].map(Op::Reduction));
    }
    if !scalar.starts_with('u') {
        result.extend(["neg", "abs"].map(Op::Method));
    }
    if scalar.starts_with('i') {
        result.extend(["saturating_abs", "saturating_neg"].map(Op::Method));
    }
    result
}

pub struct Step {
    pub op: Op,
    pub left: usize,
    pub right: usize,
    pub lane: usize,
    pub indices: Vec<usize>,
}

pub fn steps(seed: u64, scalar: &str, lanes: usize) -> Vec<Vec<Step>> {
    let mut random = Random::new(seed);
    let mut operations = operations(scalar);
    for _ in 0..8 {
        operations.push(operations[random.below(operations.len())].clone());
    }
    random.shuffle(&mut operations);
    operations
        .chunks(10)
        .map(|operations| {
            operations
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, op)| {
                    let count = if matches!(op, Op::Swizzle(true)) {
                        lanes * 2
                    } else {
                        lanes
                    };
                    Step {
                        op,
                        left: if random.below(2) == 0 {
                            index + 1
                        } else {
                            random.below(index + 2)
                        },
                        right: random.below(index + 2),
                        lane: random.below(lanes),
                        indices: (0..lanes).map(|_| random.below(count)).collect(),
                    }
                })
                .collect()
        })
        .collect()
}

pub fn inputs(seed: u64, scalar: &str) -> Vec<[u32; 4]> {
    let mut random = Random::new(seed);
    let mut values = (0..129)
        .map(|_| std::array::from_fn(|_| random.next() as u32))
        .collect::<Vec<_>>();
    for bits in [
        0_u32,
        1,
        0x7f,
        0x80,
        0xff,
        0x7fff,
        0x8000,
        0xffff,
        0x7fff_ffff,
        0x8000_0000,
        0x7fc0_1234,
        0x7f80_0000,
        0xff80_0000,
        u32::MAX,
    ] {
        values.push([bits; 4]);
        values.push([bits, !bits, bits.wrapping_add(1), bits.wrapping_sub(1)]);
    }
    if scalar == "f32" {
        for word in values.iter_mut().flatten() {
            if !matches!(
                *word,
                0 | 0x8000_0000 | 0x7fc0_1234 | 0x7f80_0000 | 0xff80_0000
            ) {
                *word = (((*word % 257) as f32 - 128.0) / 4.0).to_bits();
            }
        }
    }
    values
}
