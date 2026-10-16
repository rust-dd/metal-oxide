use super::Random;
use metal_oxide::{BF16, F16};
use std::{collections::BTreeSet, fmt::Write};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Half,
    Bfloat,
    Primitive,
}

const KINDS: [Kind; 3] = [Kind::Half, Kind::Bfloat, Kind::Primitive];

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Self::Half => "F16",
            Self::Bfloat => "BF16",
            Self::Primitive => "f16",
        }
    }

    fn round(self, value: f64) -> u16 {
        match self {
            Self::Half | Self::Primitive => F16::from_f64(value).to_bits(),
            Self::Bfloat => BF16::from_f64(value).to_bits(),
        }
    }

    fn widen(self, bits: u16) -> f64 {
        match self {
            Self::Half | Self::Primitive => F16::from_bits(bits).to_f64(),
            Self::Bfloat => BF16::from_bits(bits).to_f64(),
        }
    }

    fn subnormal(self, bits: u16) -> bool {
        let magnitude = bits & 0x7fff;
        magnitude != 0
            && magnitude
                < match self {
                    Self::Half | Self::Primitive => 0x0400,
                    Self::Bfloat => 0x0080,
                }
    }

    fn narrow_source(self, value: &str) -> String {
        match self {
            Self::Primitive => format!("({value}) as f16"),
            _ => format!("{}::from_f32({value})", self.name()),
        }
    }

    fn widen_source(self, index: usize) -> String {
        match self {
            Self::Primitive => format!("v{index} as f32"),
            _ => format!("v{index}.to_f32()"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    FromBits,
    FromFloat,
    CopyBits,
    Convert,
    Negate,
    Binary(&'static str, bool),
    Compare(&'static str),
    PartialCompare,
}

const OPS: [Op; 21] = [
    Op::FromBits,
    Op::FromFloat,
    Op::CopyBits,
    Op::Convert,
    Op::Negate,
    Op::Binary("+", false),
    Op::Binary("-", false),
    Op::Binary("*", false),
    Op::Binary("/", false),
    Op::Binary("%", false),
    Op::Binary("+", true),
    Op::Binary("-", true),
    Op::Binary("*", true),
    Op::Binary("/", true),
    Op::Binary("%", true),
    Op::Compare("=="),
    Op::Compare("!="),
    Op::Compare("<"),
    Op::Compare("<="),
    Op::Compare(">"),
    Op::Compare(">="),
];

#[derive(Debug)]
struct Step {
    kind: Kind,
    op: Op,
    left: usize,
    right: usize,
    word: usize,
}

pub struct Program {
    steps: Vec<Step>,
}

impl Program {
    pub fn generate(seed: u64) -> Self {
        let mut random = Random::new(seed);
        let mut steps = Vec::new();
        for kind in KINDS {
            for (op, word) in [(Op::FromBits, 0), (Op::FromFloat, 1)] {
                steps.push(Step {
                    kind,
                    op,
                    left: 0,
                    right: 0,
                    word: word + if kind == Kind::Bfloat { 2 } else { 0 },
                });
            }
        }
        let mut operations = KINDS
            .into_iter()
            .flat_map(|kind| {
                OPS.into_iter()
                    .chain([Op::PartialCompare])
                    .map(move |op| (kind, op))
            })
            .collect::<Vec<_>>();
        for _ in 0..12 {
            operations.push((
                KINDS[random.below(KINDS.len())],
                OPS[random.below(OPS.len())],
            ));
        }
        random.shuffle(&mut operations);
        for (kind, op) in operations {
            let candidates = steps
                .iter()
                .enumerate()
                .filter_map(|(index, step)| (step.kind == kind).then_some(index))
                .collect::<Vec<_>>();
            let left = if matches!(op, Op::Convert) {
                let other = steps
                    .iter()
                    .enumerate()
                    .filter_map(|(index, step)| (step.kind != kind).then_some(index))
                    .collect::<Vec<_>>();
                other[random.below(other.len())]
            } else if random.below(2) == 0 {
                *candidates.last().unwrap()
            } else {
                candidates[random.below(candidates.len())]
            };
            steps.push(Step {
                kind,
                op,
                left,
                right: candidates[random.below(candidates.len())],
                word: random.below(4),
            });
        }
        Self { steps }
    }

    pub fn words(&self) -> usize {
        self.steps.len() * 2
    }

    pub fn coverage(&self) -> BTreeSet<String> {
        self.steps
            .iter()
            .map(|step| format!("{}:{:?}", step.kind.name(), step.op))
            .collect()
    }

    pub fn source(&self) -> String {
        let mut source = String::from(
            "#![no_std]\n#![feature(f16)]\nuse metal_oxide_device::{kernel, F16, BF16, ReadBuffer, WriteBuffer, block_idx, block_dim, thread_idx};\n#[kernel]\npub unsafe fn generated(input: ReadBuffer<[u32; 4]>, output: WriteBuffer<u32>, n: u32) {\n    let i = block_idx().x * block_dim().x + thread_idx().x;\n    if i < n { unsafe {\n        let words = input.load_unchecked(i);\n",
        );
        for (index, step) in self.steps.iter().enumerate() {
            let Step {
                kind,
                op,
                left,
                right,
                word,
            } = step;
            let ty = kind.name();
            let value = match op {
                Op::FromBits => format!("{ty}::from_bits(words[{word}] as u16)"),
                Op::FromFloat => kind.narrow_source(&format!("f32::from_bits(words[{word}])")),
                Op::CopyBits => format!("{ty}::from_bits(v{left}.to_bits())"),
                Op::Convert => kind.narrow_source(&self.steps[*left].kind.widen_source(*left)),
                Op::Negate => format!("-v{left}"),
                Op::Binary(symbol, false) => format!("v{left} {symbol} v{right}"),
                Op::Binary(symbol, true) => {
                    writeln!(
                        source,
                        "        let mut v{index} = v{left};\n        v{index} {symbol}= v{right};"
                    )
                    .unwrap();
                    String::new()
                }
                Op::Compare(symbol) => {
                    kind.narrow_source(&format!("if v{left} {symbol} v{right} {{ 1.0_f32 }} else {{ 0.0_f32 }}"))
                }
                Op::PartialCompare => kind.narrow_source(&format!(
                    "match v{left}.partial_cmp(&v{right}) {{ Some(core::cmp::Ordering::Less) => -1.0_f32, Some(core::cmp::Ordering::Equal) => 0.0_f32, Some(core::cmp::Ordering::Greater) => 1.0_f32, None => 2.0_f32 }}"
                )),
            };
            if !value.is_empty() {
                writeln!(source, "        let v{index} = {value};").unwrap();
            }
            writeln!(source, "        output.store_unchecked(i * {} + {}, v{index}.to_bits() as u32);\n        output.store_unchecked(i * {} + {}, ({}).to_bits());", self.words(), index * 2, self.words(), index * 2 + 1, kind.widen_source(index)).unwrap();
        }
        source.push_str("    } }\n}\n");
        source
    }

    pub fn verify(&self, input: [u32; 4], output: &[u32]) -> Result<(), String> {
        if output.len() != self.words() {
            return Err(format!("{} words, expected {}", output.len(), self.words()));
        }
        let mut values = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            let bits = u16::try_from(output[index * 2])
                .map_err(|_| format!("step {index} returned more than 16 bits"))?;
            let expected = self.expected(step, input, &values);
            let preserves_bits = matches!(step.op, Op::FromBits | Op::CopyBits | Op::Negate);
            let same = |expected| {
                bits == expected
                    || (!preserves_bits
                        && step.kind.widen(bits).is_nan()
                        && step.kind.widen(expected).is_nan())
            };
            if !expected.iter().copied().any(same) {
                return Err(format!(
                    "step {index} {step:?}: {bits:04x}, expected {expected:04x?}"
                ));
            }
            let wide = f32::from_bits(output[index * 2 + 1]);
            let expected_wide = step.kind.widen(bits) as f32;
            if wide.to_bits() != expected_wide.to_bits()
                && !(wide.is_nan() && expected_wide.is_nan())
            {
                return Err(format!(
                    "step {index} widened {bits:04x} to {:08x}, expected {:08x}",
                    wide.to_bits(),
                    expected_wide.to_bits()
                ));
            }
            // Each checked intermediate is a valid operand even when Metal flushes a subnormal.
            values.push(bits);
        }
        Ok(())
    }

    fn expected(&self, step: &Step, input: [u32; 4], values: &[u16]) -> Vec<u16> {
        let Step {
            kind,
            op,
            left,
            right,
            word,
        } = *step;
        match op {
            Op::FromBits => return vec![input[word] as u16],
            Op::FromFloat => return vec![kind.round(f64::from(f32::from_bits(input[word])))],
            Op::CopyBits => return vec![values[left]],
            Op::Convert => return vec![kind.round(self.steps[left].kind.widen(values[left]))],
            Op::Negate => return vec![values[left] ^ 0x8000],
            _ => {}
        }
        let a = kind.widen(values[left]);
        let b = kind.widen(values[right]);
        if let Op::Compare(symbol) = op {
            let value = match symbol {
                "==" => a == b,
                "!=" => a != b,
                "<" => a < b,
                "<=" => a <= b,
                ">" => a > b,
                ">=" => a >= b,
                _ => unreachable!(),
            };
            return vec![kind.round(f64::from(value))];
        }
        if matches!(op, Op::PartialCompare) {
            return vec![kind.round(match a.partial_cmp(&b) {
                Some(std::cmp::Ordering::Less) => -1.0,
                Some(std::cmp::Ordering::Equal) => 0.0,
                Some(std::cmp::Ordering::Greater) => 1.0,
                None => 2.0,
            })];
        }
        let Op::Binary(symbol, _) = op else {
            unreachable!()
        };
        let alternatives = |value, bits| {
            [
                value,
                if kind.subnormal(bits) {
                    0.0_f64.copysign(value)
                } else {
                    value
                },
            ]
        };
        let mut expected = Vec::new();
        for a in alternatives(a, values[left]) {
            for b in alternatives(b, values[right]) {
                let bits = kind.round(match symbol {
                    "+" => a + b,
                    "-" => a - b,
                    "*" => a * b,
                    "/" => a / b,
                    "%" => a % b,
                    _ => unreachable!(),
                });
                expected.push(bits);
                if kind.subnormal(bits) {
                    expected.push(bits & 0x8000);
                }
            }
        }
        expected
    }
}

pub fn inputs(seed: u64) -> Vec<[u32; 4]> {
    let mut random = Random::new(seed);
    let mut inputs = (0..257)
        .map(|_| {
            [
                random.next() as u32 & 0xffff,
                ((random.below(8193) as f32 - 4096.0) / 64.0).to_bits(),
                random.next() as u32 & 0xffff,
                ((random.below(8193) as f32 - 4096.0) / 64.0).to_bits(),
            ]
        })
        .collect::<Vec<_>>();
    for bits in [
        0, 1, 0x007f, 0x0080, 0x03ff, 0x0400, 0x3c00, 0x3f80, 0x7bff, 0x7c00, 0x7c01, 0x7f7f,
        0x7f80, 0x7fc1, 0x8000, 0x8001, 0xfc00, 0xff80,
    ] {
        for float in [
            0_u32,
            0x8000_0000,
            1,
            0x3380_0000,
            0x3f80_8000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_1234,
        ] {
            inputs.push([bits, float, bits, float]);
        }
    }
    inputs
}
