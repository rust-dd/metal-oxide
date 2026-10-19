pub mod model;
mod printer;

use crate::{generated_support, support};
use model::Step;
use printer::Printer;
use std::{collections::BTreeSet, fmt::Write, process::Command, time::Duration};

pub struct Program {
    pub scalar: &'static str,
    pub lanes: usize,
    pub seed: u64,
    pub inputs: Vec<[u32; 4]>,
    steps: Vec<Vec<Step>>,
    printed: Vec<Printer>,
}

impl Program {
    pub fn generate(seed: u64, scalar: &'static str, lanes: usize) -> Self {
        let steps = model::steps(seed, scalar, lanes);
        let printed = steps
            .iter()
            .enumerate()
            .map(|(index, steps)| {
                let mut printed = Printer::program(scalar, lanes, steps);
                for word in &mut printed.words {
                    word.operation = format!("graph{index}:{}", word.operation);
                }
                printed
            })
            .collect();
        Self {
            scalar,
            lanes,
            seed,
            inputs: model::inputs(seed, scalar),
            steps,
            printed,
        }
    }

    pub fn name(&self, phase: &str) -> String {
        format!(
            "vectors-{phase}-{}-{}-{:016x}",
            self.scalar, self.lanes, self.seed
        )
    }

    pub fn words(&self) -> usize {
        self.printed.iter().map(|printed| printed.words.len()).sum()
    }

    pub fn coverage(&self) -> BTreeSet<String> {
        self.steps
            .iter()
            .flatten()
            .map(|step| format!("{:?}", step.op))
            .collect()
    }

    pub fn kernels(&self) -> impl Iterator<Item = String> {
        (0..self.printed.len()).map(|index| format!("generated{index}"))
    }

    pub fn source(&self) -> String {
        let words = self.words();
        let inputs = self
            .inputs
            .iter()
            .map(|input| format!("{input:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut source = String::from(
            "#![cfg_attr(not(cpu_reference), no_std)]\n#![feature(portable_simd)]\nuse core::simd::prelude::*;\n#[cfg(not(cpu_reference))]\nuse metal_oxide_device::{kernel, ReadBuffer, WriteBuffer, block_idx, block_dim, thread_idx};\n",
        );
        let mut offset = 0;
        let mut calls = String::new();
        for (index, printed) in self.printed.iter().enumerate() {
            let count = printed.words.len();
            writeln!(source, "#[cfg(cpu_reference)]\nfn run{index}(words: [u32; 4]) -> [u32; {count}] {{\n    let mut output = [0_u32; {count}];\n    macro_rules! record {{ ($slot:expr, $value:expr) => {{ output[$slot] = $value }}; }}\n{}\n    output\n}}\n#[cfg(not(cpu_reference))]\nunsafe fn run{index}(words: [u32; 4], output: WriteBuffer<u32>, base: u32) {{\n    macro_rules! record {{ ($slot:expr, $value:expr) => {{ unsafe {{ output.store_unchecked(base + $slot, $value) }} }}; }}\n{}\n}}\n#[cfg(not(cpu_reference))]\n#[kernel]\npub unsafe fn generated{index}(input: ReadBuffer<[u32; 4]>, output: WriteBuffer<u32>, n: u32) {{\n    let i = block_idx().x * block_dim().x + thread_idx().x;\n    if i < n {{ unsafe {{\n        run{index}(input.load_unchecked(i), output, i * {words} + {offset});\n    }} }}\n}}\n", printed.text, printed.text).unwrap();
            writeln!(
                calls,
                "        for word in run{index}(input) {{ print!(\"{{word}} \"); }}"
            )
            .unwrap();
            offset += count;
        }
        writeln!(source, "#[cfg(cpu_reference)]\nfn main() {{\n    let inputs: [[u32; 4]; {}] = [{inputs}];\n    for input in inputs {{\n{calls}        println!();\n    }}\n}}", self.inputs.len()).unwrap();
        source
    }

    pub fn cpu(&self, name: &str) -> Vec<Vec<u32>> {
        let directory = generated_support::directory(name);
        let source = directory.join("case.rs");
        std::fs::write(&source, self.source()).unwrap();
        let binary = directory.join("cpu");
        let mut rustc = Command::new("rustc");
        rustc
            .arg(&source)
            .args([
                "--edition=2024",
                "--cfg",
                "cpu_reference",
                "-O",
                "-C",
                "overflow-checks=off",
            ])
            .arg("-o")
            .arg(&binary);
        let output = support::process::output(&mut rustc, Duration::from_secs(60)).unwrap();
        support::succeeded(output).unwrap_or_else(|error| panic!("{}: {error}", source.display()));
        let output = support::process::output(
            &mut Command::new(&binary),
            support::process::EXECUTION_TIMEOUT,
        )
        .unwrap();
        let output = support::checked(output);
        std::fs::write(directory.join("cpu.stdout"), &output).unwrap();
        let rows = output
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|word| word.parse::<u32>().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), self.inputs.len());
        assert!(rows.iter().all(|row| row.len() == self.words()));
        rows
    }

    pub fn mismatch(&self, expected: &[u32], actual: &[u32]) -> Option<String> {
        if expected.len() != self.words() || actual.len() != self.words() {
            return Some(format!(
                "expected {} words, got {} and {}",
                self.words(),
                expected.len(),
                actual.len()
            ));
        }
        expected
            .iter()
            .zip(actual)
            .zip(self.printed.iter().flat_map(|printed| &printed.words))
            .enumerate()
            .find_map(|(word, ((&expected, &actual), kind))| {
                let a = f32::from_bits(actual);
                let e = f32::from_bits(expected);
                let equal = actual == expected
                    || (kind.float && ((a.is_nan() && e.is_nan()) || (a == 0.0 && e == 0.0)));
                (!equal).then(|| {
                    format!(
                        "word {word} {}: {actual:08x}, expected {expected:08x}",
                        kind.operation
                    )
                })
            })
    }
}

pub fn programs() -> impl Iterator<Item = Program> {
    model::SEEDS.into_iter().flat_map(|seed| {
        model::SCALARS.into_iter().flat_map(move |scalar| {
            (2..=4).map(move |lanes| Program::generate(seed, scalar, lanes))
        })
    })
}
