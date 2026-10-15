//! Seeded programs of the supported subset whose words the CPU, the host MSL and the GPU must
//! agree on; in cooperative programs, a block's threads run `cooperate` together first.

mod ast;
mod borrows;
mod builder;
mod closures;
mod cooperation;
mod entry;
mod expressions;
mod iterators;
pub mod model;
mod random;
mod scope;
mod shrink;
mod types;
mod variants;

use ast::{Function, Printer};
use builder::Builder;
use types::{Enum, Record};

/// How the oracle compares one output word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Word {
    /// Integers and bools, bit for bit.
    Exact,
    /// The bits of an `f32` built from exact operations: equal bits, or NaN on both sides.
    Float,
}

#[derive(Clone)]
pub struct Program {
    pub seed: u64,
    /// Whether the program builds with `-C overflow-checks=on`.
    pub overflow_checks: bool,
    records: Vec<Record>,
    enums: Vec<Enum>,
    helpers: Vec<Function>,
    /// What the threads of a block compute together before `run`.
    cooperation: Option<Function>,
    run: Function,
    pub words: Vec<Word>,
}

impl Program {
    pub fn generate(seed: u64) -> Self {
        let mut builder = Builder::new(seed);
        builder.fallible = builder.random.chance(1, 2);
        let overflow_checks = builder.random.chance(1, 2);
        for _ in 0..builder.random.below(5) {
            if builder.random.chance(1, 2) {
                builder.record();
            } else {
                builder.enumeration();
            }
        }
        for _ in 0..builder.random.below(4) {
            builder.helper();
        }
        let values = builder
            .random
            .chance(1, 3)
            .then(|| 1 + builder.random.below(3) as u32);
        let cooperation = values.map(|values| builder.cooperation(values));
        let words = 1 + builder.random.below(4) as u32;
        let (run, words) = builder.run(words, values.unwrap_or(0));
        Self {
            seed,
            overflow_checks,
            records: builder.records,
            enums: builder.enums,
            helpers: builder.helpers,
            cooperation,
            run,
            words,
        }
    }

    pub fn cooperative(&self) -> bool {
        self.cooperation.is_some()
    }

    /// The element type of the device counters that `cooperate` counts in, if it does.
    pub fn counters(&self) -> Option<String> {
        let parameters = self
            .cooperation
            .iter()
            .flat_map(|function| &function.parameters);
        parameters.map(|(_, ty)| ty).find_map(|ty| match ty {
            types::Type::Atomic(int) => Some(int.name()),
            _ => None,
        })
    }

    /// The records and enums that `functions` names, directly or through other definitions.
    fn definitions(&self, functions: &str) -> String {
        let records = self.records.iter().map(|record| {
            let fields = record
                .fields
                .iter()
                .map(|(name, ty)| format!("{name}: {}", types::render(ty, &self.records)))
                .collect::<Vec<_>>();
            let text = format!("struct {} {{ {} }}", record.name, fields.join(", "));
            (record.name.clone(), text)
        });
        let enums = self.enums.iter().enumerate().map(|(index, definition)| {
            let variants = definition
                .variants
                .iter()
                .enumerate()
                .map(|(variant, payload)| match payload.as_slice() {
                    [] => format!("V{variant}"),
                    payload => {
                        let payload = payload
                            .iter()
                            .map(|ty| types::render(ty, &self.records))
                            .collect::<Vec<_>>();
                        format!("V{variant}({})", payload.join(", "))
                    }
                })
                .collect::<Vec<_>>();
            let name = format!("E{index}");
            let text = format!("enum {name} {{ {} }}", variants.join(", "));
            (name, text)
        });
        let definitions = records.chain(enums).collect::<Vec<_>>();
        let mut used = vec![false; definitions.len()];
        let mut named = functions.to_owned();
        while let Some(index) = (0..definitions.len())
            .find(|&index| !used[index] && mentions(&named, &definitions[index].0))
        {
            used[index] = true;
            named.push_str(&definitions[index].1);
        }
        definitions
            .iter()
            .zip(used)
            .filter(|(_, used)| *used)
            .map(|((_, text), _)| format!("#[derive(Clone, Copy)]\n{text}\n\n"))
            .collect()
    }

    /// The kernel, and with `--cfg cpu_reference` a program that prints the words of `inputs`.
    pub fn source(&self, inputs: &[u32]) -> String {
        let mut printer = Printer {
            records: &self.records,
            text: String::new(),
        };
        let words = self
            .words
            .iter()
            .map(|word| match word {
                Word::Exact => "exact",
                Word::Float => "float",
            })
            .collect::<Vec<_>>();
        let checks = if self.overflow_checks { "on" } else { "off" };
        printer.text.push_str(&format!(
            "// words: {}\n// overflow checks: {checks}\n",
            words.join(" ")
        ));
        if self.cooperative() {
            printer.text.push_str("// cooperative\n");
        }
        if let Some(counters) = self.counters() {
            printer.text.push_str(&format!("// counters: {counters}\n"));
        }
        printer.text.push_str(
            "#![cfg_attr(not(cpu_reference), no_std)]\n#![allow(unused_mut, unused_variables, unused_assignments, unused_parens, unused_labels, unused_braces, unused_comparisons, unused_imports, dead_code, unreachable_code, arithmetic_overflow, unconditional_panic)]\n\n",
        );
        let mut functions = Printer {
            records: &self.records,
            text: String::new(),
        };
        for function in self.helpers.iter().chain(&self.cooperation) {
            functions.function(function);
            functions.text.push('\n');
        }
        functions.function(&self.run);
        printer.text.push_str(&self.definitions(&functions.text));
        printer.text.push_str(&functions.text);
        let inputs = inputs
            .iter()
            .map(|input| format!("{input}_u32"))
            .collect::<Vec<_>>()
            .join(", ");
        let entry = match &self.cooperation {
            Some(cooperation) => {
                let counters = self.counters();
                entry::cooperative(cooperation, counters.as_deref(), self.words.len(), &inputs)
            }
            None => entry::independent(self.words.len(), &inputs),
        };
        printer.text.push_str(&entry);
        printer.text
    }
}

/// Whether `text` names `name` as a whole word.
fn mentions(text: &str, name: &str) -> bool {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_');
    text.match_indices(name).any(|(at, _)| {
        !word(text[..at].chars().next_back()) && !word(text[at + name.len()..].chars().next())
    })
}
