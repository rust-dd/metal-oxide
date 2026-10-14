//! Seeded programs from the supported Rust subset, for differential tests: every thread runs
//! `run(input[i])` and writes its words, which the CPU, the host-compiled MSL and the GPU
//! must agree on.

mod ast;
mod builder;
mod expressions;
mod random;
mod types;

use ast::{Function, Printer};
use builder::Builder;
use types::Record;

/// How the oracle compares one output word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Word {
    /// Integers and bools, bit for bit.
    Exact,
    /// The bits of an `f32` built from exact operations: equal bits, or NaN on both sides.
    Float,
}

pub struct Program {
    pub seed: u64,
    /// Whether the program builds with `-C overflow-checks=on`.
    pub overflow_checks: bool,
    records: Vec<Record>,
    helpers: Vec<Function>,
    run: Function,
    pub words: Vec<Word>,
}

impl Program {
    pub fn generate(seed: u64) -> Self {
        let mut builder = Builder::new(seed);
        builder.fallible = builder.random.chance(1, 2);
        let overflow_checks = builder.random.chance(1, 2);
        for _ in 0..builder.random.below(3) {
            builder.record();
        }
        for _ in 0..builder.random.below(4) {
            builder.helper();
        }
        let words = 1 + builder.random.below(4) as u32;
        let (run, words) = builder.run(words);
        Self {
            seed,
            overflow_checks,
            records: builder.records,
            helpers: builder.helpers,
            run,
            words,
        }
    }

    /// The kernel, and with `--cfg cpu_reference` a program that prints the words of `inputs`.
    pub fn source(&self, inputs: &[u32]) -> String {
        let mut printer = Printer {
            records: &self.records,
            text: String::new(),
        };
        printer.text.push_str(
            "#![cfg_attr(not(cpu_reference), no_std)]\n#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]\n\n",
        );
        for record in &self.records {
            let fields = record
                .fields
                .iter()
                .map(|(name, ty)| format!("{name}: {}", types::render(ty, &self.records)))
                .collect::<Vec<_>>();
            printer.text.push_str(&format!(
                "#[derive(Clone, Copy)]\nstruct {} {{ {} }}\n\n",
                record.name,
                fields.join(", ")
            ));
        }
        for helper in &self.helpers {
            printer.function(helper);
            printer.text.push('\n');
        }
        printer.function(&self.run);
        let words = self.words.len();
        let stores = (0..words)
            .map(|word| {
                format!("        output.store_unchecked(i * {words} + {word}, words[{word}]);\n")
            })
            .collect::<String>();
        let inputs = inputs
            .iter()
            .map(|input| format!("{input}_u32"))
            .collect::<Vec<_>>();
        printer.text.push_str(&format!(
            r#"
#[cfg(not(cpu_reference))]
use metal_oxide_device::{{ReadBuffer, WriteBuffer, block_dim, block_idx, kernel, thread_idx}};

#[cfg(not(cpu_reference))]
#[kernel]
pub unsafe fn generated(input: ReadBuffer<u32>, output: WriteBuffer<u32>, n: u32) {{
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {{
        let words = run(unsafe {{ input.load_unchecked(i) }});
        unsafe {{
{stores}        }}
    }}
}}

#[cfg(cpu_reference)]
fn main() {{
    let location = std::sync::Arc::new(std::sync::Mutex::new((0, 0)));
    let recorded = std::sync::Arc::clone(&location);
    std::panic::set_hook(Box::new(move |info| {{
        let panicked = info.location().unwrap();
        *recorded.lock().unwrap() = (panicked.line(), panicked.column());
    }}));
    for input in [{inputs}] {{
        match std::panic::catch_unwind(|| run(input)) {{
            Ok(words) => println!("{{}}", words.map(|word| word.to_string()).join(" ")),
            Err(_) => {{
                let (line, column) = *location.lock().unwrap();
                println!("panic {{line}}:{{column}}");
            }}
        }}
    }}
}}
"#,
            inputs = inputs.join(", ")
        ));
        printer.text
    }
}
