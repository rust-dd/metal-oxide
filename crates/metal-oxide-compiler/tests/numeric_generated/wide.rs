use super::Random;
use std::fmt::Write;

enum Transform {
    Add(u64),
    Xor(u64),
    Rotate(u32),
}

struct Step {
    minimum: bool,
    transform: Transform,
}

pub struct Program {
    steps: Vec<Step>,
}

impl Program {
    pub fn generate(seed: u64) -> Self {
        let mut random = Random::new(seed);
        let mut directions = [false, true].repeat(8);
        random.shuffle(&mut directions);
        let steps = directions
            .into_iter()
            .map(|minimum| {
                let transform = match random.below(3) {
                    0 => Transform::Add(random.next()),
                    1 => Transform::Xor(random.next()),
                    _ => Transform::Rotate(random.below(64) as u32),
                };
                Step { minimum, transform }
            })
            .collect();
        Self { steps }
    }

    pub fn source(&self) -> String {
        let mut source = String::from(
            "#![no_std]\nuse core::sync::atomic::Ordering::Relaxed;\nuse metal_oxide_device::{kernel, AtomicBuffer, ReadBuffer, block_idx, block_dim, thread_idx};\n#[kernel]\npub unsafe fn generated(input: ReadBuffer<u64>, cell: AtomicBuffer<u64>, n: u32) {\n    let i = block_idx().x * block_dim().x + thread_idx().x;\n    if i < n { unsafe {\n        let v0 = input.load_unchecked(i);\n",
        );
        for (index, step) in self.steps.iter().enumerate() {
            let expression = match step.transform {
                Transform::Add(value) => format!("v{index}.wrapping_add({value}_u64)"),
                Transform::Xor(value) => format!("v{index} ^ {value}_u64"),
                Transform::Rotate(shift) => format!("v{index}.rotate_left({shift})"),
            };
            let next = index + 1;
            let (operation, cell) = if step.minimum { ("min", 0) } else { ("max", 1) };
            writeln!(source, "        let v{next} = {expression};\n        let () = cell.{operation}({cell}, v{next}, Relaxed);").unwrap();
        }
        source.push_str("    } }\n}\n");
        source
    }

    pub fn reference(&self, inputs: &[u64]) -> [u64; 2] {
        let mut result = [u64::MAX, 0];
        for &input in inputs {
            let mut value = input;
            for step in &self.steps {
                value = match step.transform {
                    Transform::Add(addend) => value.wrapping_add(addend),
                    Transform::Xor(mask) => value ^ mask,
                    Transform::Rotate(shift) => value.rotate_left(shift),
                };
                if step.minimum {
                    result[0] = result[0].min(value);
                } else {
                    result[1] = result[1].max(value);
                }
            }
        }
        result
    }
}

pub fn inputs(seed: u64) -> Vec<u64> {
    let mut random = Random::new(seed);
    [
        0,
        1,
        u64::from(u32::MAX),
        1 << 32,
        1 << 48,
        1 << 63,
        u64::MAX,
    ]
    .into_iter()
    .chain((0..4097).map(|_| random.next()))
    .collect()
}
