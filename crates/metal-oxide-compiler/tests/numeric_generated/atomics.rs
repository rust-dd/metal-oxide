use super::Random;
use std::{collections::BTreeSet, fmt::Write};

#[derive(Clone, Copy, Debug)]
enum Op {
    Load,
    Store,
    Exchange,
    Add,
    Sub,
    Compare(bool),
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Self::Load => "load",
            Self::Store => "store",
            Self::Exchange => "exchange",
            Self::Add => "fetch_add",
            Self::Sub => "fetch_sub",
            Self::Compare(_) => "compare_exchange_weak",
        }
    }
}

#[derive(Debug)]
struct Step {
    op: Op,
    order: &'static str,
    failure: &'static str,
    argument: Argument,
}

#[derive(Debug)]
enum Argument {
    Input(usize),
    Previous(usize),
}

pub struct FloatProgram {
    steps: Vec<Step>,
}

impl FloatProgram {
    pub fn generate(seed: u64, ordered: bool) -> Self {
        let mut random = Random::new(seed);
        let mut operations = vec![
            (Op::Load, "Relaxed", "Relaxed"),
            (Op::Store, "Relaxed", "Relaxed"),
            (Op::Exchange, "Relaxed", "Relaxed"),
            (Op::Add, "Relaxed", "Relaxed"),
            (Op::Sub, "Relaxed", "Relaxed"),
            (Op::Compare(true), "Relaxed", "Relaxed"),
            (Op::Compare(false), "Relaxed", "Relaxed"),
        ];
        if ordered {
            operations.extend([
                (Op::Load, "Acquire", "Relaxed"),
                (Op::Store, "Release", "Relaxed"),
            ]);
            for op in [Op::Exchange, Op::Add, Op::Sub] {
                for order in ["Acquire", "Release", "AcqRel"] {
                    operations.push((op, order, "Relaxed"));
                }
            }
            for (success, failure) in [
                ("Acquire", "Relaxed"),
                ("Acquire", "Acquire"),
                ("Release", "Relaxed"),
                ("AcqRel", "Relaxed"),
                ("AcqRel", "Acquire"),
            ] {
                operations.push((Op::Compare(random.below(2) == 0), success, failure));
            }
        } else {
            for _ in 0..12 {
                operations.push(operations[random.below(7)]);
            }
        }
        random.shuffle(&mut operations);
        let steps = operations
            .into_iter()
            .enumerate()
            .map(|(index, (op, order, failure))| Step {
                op,
                order,
                failure,
                argument: if index != 0 && random.below(2) == 0 {
                    Argument::Previous(random.below(index))
                } else {
                    Argument::Input(random.below(4))
                },
            })
            .collect();
        Self { steps }
    }

    pub fn words(&self) -> usize {
        self.steps.len() * 3
    }

    pub fn coverage(&self) -> BTreeSet<String> {
        self.steps
            .iter()
            .map(|step| format!("{}:{}:{}", step.op.name(), step.order, step.failure))
            .collect()
    }

    pub fn source(&self) -> String {
        let mut source = String::from(
            "#![no_std]\nuse core::sync::atomic::Ordering::*;\nuse metal_oxide_device::{kernel, AtomicBuffer, ReadBuffer, WriteBuffer, block_idx, block_dim, thread_idx};\n#[kernel]\npub unsafe fn generated(input: ReadBuffer<[u32; 4]>, cell: AtomicBuffer<f32>, output: WriteBuffer<u32>, n: u32) {\n    let i = block_idx().x * block_dim().x + thread_idx().x;\n    if i < n { unsafe {\n        let words = input.load_unchecked(i);\n",
        );
        for (index, step) in self.steps.iter().enumerate() {
            let argument = match step.argument {
                Argument::Input(word) => format!("f32::from_bits(words[{word}])"),
                Argument::Previous(previous) => format!("v{previous}"),
            };
            let order = step.order;
            match step.op {
                Op::Compare(matching) => {
                    let expected = if matching {
                        "cell.load(i, Relaxed)"
                    } else {
                        "f32::from_bits(cell.load(i, Relaxed).to_bits() ^ 0x80000000)"
                    };
                    writeln!(source, "        let expected{index}: f32 = {expected};\n        let (v{index}, ok{index}) = cell.compare_exchange_weak(i, expected{index}, {argument}, {order}, {});", step.failure).unwrap();
                }
                Op::Store => {
                    writeln!(source, "        cell.store(i, {argument}, {order});\n        let v{index} = 0.0_f32;\n        let ok{index} = false;").unwrap();
                }
                op => {
                    let args = if matches!(op, Op::Load) {
                        String::new()
                    } else {
                        format!("{argument}, ")
                    };
                    writeln!(source, "        let v{index} = cell.{}(i, {args}{order});\n        let ok{index} = false;", op.name()).unwrap();
                }
            }
            for (word, expression) in [
                format!("v{index}.to_bits()"),
                format!("ok{index} as u32"),
                "cell.load(i, Relaxed).to_bits()".into(),
            ]
            .into_iter()
            .enumerate()
            {
                writeln!(
                    source,
                    "        output.store_unchecked(i * {} + {}, {expression});",
                    self.words(),
                    index * 3 + word
                )
                .unwrap();
            }
        }
        source.push_str("    } }\n}\n");
        source
    }

    pub fn verify(&self, input: [u32; 4], output: &[u32], final_bits: u32) -> Result<(), String> {
        if output.len() != self.words() {
            return Err(format!("{} words, expected {}", output.len(), self.words()));
        }
        let mut state = input[0];
        let mut previous = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            let argument = match step.argument {
                Argument::Input(word) => input[word],
                Argument::Previous(index) => previous[index],
            };
            let actual = &output[index * 3..index * 3 + 3];
            let expected_old = if matches!(step.op, Op::Store) {
                0
            } else {
                state
            };
            if actual[0] != expected_old {
                return Err(format!(
                    "step {index} {step:?}: old={:08x}, expected {expected_old:08x}",
                    actual[0]
                ));
            }
            if actual[1] > 1 || (!matches!(step.op, Op::Compare(true)) && actual[1] != 0) {
                return Err(format!(
                    "step {index} {step:?}: invalid success flag {}",
                    actual[1]
                ));
            }
            let next = match step.op {
                Op::Load => state,
                Op::Store | Op::Exchange => argument,
                Op::Add => (f32::from_bits(state) + f32::from_bits(argument)).to_bits(),
                Op::Sub => (f32::from_bits(state) - f32::from_bits(argument)).to_bits(),
                Op::Compare(_) => {
                    if actual[1] == 1 {
                        argument
                    } else {
                        state
                    }
                }
            };
            let nan_arithmetic = matches!(step.op, Op::Add | Op::Sub)
                && f32::from_bits(next).is_nan()
                && f32::from_bits(actual[2]).is_nan();
            if actual[2] != next && !nan_arithmetic {
                return Err(format!(
                    "step {index} {step:?}: after={:08x}, expected {next:08x}",
                    actual[2]
                ));
            }
            state = actual[2];
            previous.push(actual[0]);
        }
        if state != final_bits {
            return Err(format!(
                "final state={final_bits:08x}, expected {state:08x}"
            ));
        }
        Ok(())
    }

    pub fn verify_orders(&self, source: &str) {
        let name = |order| match order {
            "Relaxed" => "relaxed",
            "Acquire" => "acquire",
            "Release" => "release",
            "AcqRel" => "acq_rel",
            _ => unreachable!(),
        };
        for step in &self.steps {
            let order = name(step.order);
            let operation = if matches!(step.op, Op::Compare(_)) {
                format!(
                    "metal_oxide_compare_exchange_weak_{order}_{}",
                    name(step.failure)
                )
            } else {
                format!("atomic_{}_explicit", step.op.name())
            };
            assert!(
                source.lines().any(|line| line.contains(&operation)
                    && (matches!(step.op, Op::Compare(_))
                        || line.contains(&format!("memory_order_{order}")))),
                "missing {operation} with {order}"
            );
            if matches!(step.op, Op::Compare(_)) {
                assert!(source.contains(&format!(
                    "memory_order_{order}, memory_order_{}",
                    name(step.failure)
                )));
            }
        }
    }
}

pub fn inputs(seed: u64) -> Vec<[u32; 4]> {
    let mut random = Random::new(seed);
    (0..257)
        .map(|index| {
            let special = [
                0,
                0x8000_0000,
                0x7fc0_1234,
                0xffc0_5678,
                0x7f80_0000,
                0xff80_0000,
            ];
            let finite = ((random.below(2049) as f32 - 1024.0) / 4.0).to_bits();
            let initial = special.get(index % 8).copied().unwrap_or(finite);
            [
                initial,
                special[random.below(special.len())],
                finite,
                ((random.below(2049) as f32 - 1024.0) / 4.0).to_bits(),
            ]
        })
        .collect()
}
