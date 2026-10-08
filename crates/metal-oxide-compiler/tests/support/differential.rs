use std::{path::PathBuf, process::Command, time::Duration};

pub const SEEDS: [u32; 12] = [
    0,
    1,
    7,
    31,
    255,
    257,
    1023,
    65535,
    0x12345678,
    0xdeadbeef,
    0x80000000,
    u32::MAX,
];

pub struct Case {
    pub directory: PathBuf,
    pub source: PathBuf,
    pub inputs: Vec<u32>,
}

impl Case {
    pub fn new(seed: u32) -> Self {
        let directory = super::root().join(format!(
            "target/differential/{}-{seed:08x}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let mut state = seed;
        let mut inputs = vec![0, 1, u32::MAX, 0x7fffffff, 0x80000000, 65535, 65536];
        while inputs.len() < 257 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            inputs.push(state);
        }
        let source = directory.join("case.rs");
        let text = format!(
            r#"#![cfg_attr(not(cpu_reference), no_std)]
#[derive(Clone, Copy)]
struct Record {{ word: u32, lanes: [u16; 3], pair: (i32, f32) }}

fn mix<const BIAS: u32>(mut value: Record) -> Record {{
    let mut step = 0_u32;
    while step < {limit} {{
        match (value.word ^ step) & 3 {{
            0 => value.word = value.word.wrapping_add(BIAS),
            1 => value.word ^= (value.lanes[0] as i16) as u32,
            2 => value.word = value.word.wrapping_mul({factor}),
            _ => value.word = (value.word >> {shift}) | (value.word << {left}),
        }}
        step += 1;
        if step == 2 && value.word & 1 == 1 {{ continue; }}
        value.lanes[1] = value.lanes[1].wrapping_add(value.lanes[2]);
        value.word ^= (value.lanes[1] as u32) / {divisor};
        if value.word & 31 == 17 {{ break; }}
    }}
    value
}}

fn evaluate(input: u32) -> u32 {{
    let record = Record {{ word: input, lanes: [input as u16, (input >> 16) as u16, {lane}], pair: (input as i32, ((input & 1023) as f32 - 512.0) * 0.5) }};
    let result = mix::<{bias}>(record);
    result.word ^ ((result.pair.0 >> {shift}) as u32) ^ ((result.pair.1 as i16) as u32)
}}

#[cfg(not(cpu_reference))]
use metal_oxide_device::{{ReadBuffer, WriteBuffer, block_idx, block_dim, thread_idx, kernel}};
#[cfg(not(cpu_reference))]
#[kernel]
pub unsafe fn seeded(input: ReadBuffer<u32>, output: WriteBuffer<u32>, n: u32) {{
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n {{ unsafe {{ output.store_unchecked(i, evaluate(input.load_unchecked(i))); }} }}
}}

#[cfg(cpu_reference)]
fn main() {{
    for value in {inputs:?} {{ println!("{{}}", evaluate(value)); }}
}}
"#,
            limit = seed % 5 + 1,
            factor = seed | 1,
            shift = seed % 31 + 1,
            left = 31 - seed % 31,
            divisor = seed % 7 + 1,
            lane = (seed as u16),
            bias = seed.wrapping_add(17)
        );
        std::fs::write(&source, text).unwrap();
        std::fs::write(directory.join("seed.txt"), format!("{seed}\n")).unwrap();
        Self {
            directory,
            source,
            inputs,
        }
    }

    pub fn compile(&self) -> PathBuf {
        let output = self.directory.join("generated");
        super::checked(super::emit_into(
            self.source.to_str().unwrap(),
            &["-C", "overflow-checks=off"],
            &output,
        ));
        output
    }

    pub fn cpu(&self) -> Vec<u32> {
        let binary = self.directory.join("cpu");
        super::checked(
            super::process::output(
                Command::new("rustc")
                    .arg(&self.source)
                    .args([
                        "--edition=2024",
                        "--cfg",
                        "cpu_reference",
                        "-O",
                        "-C",
                        "overflow-checks=off",
                    ])
                    .arg("-o")
                    .arg(&binary),
                Duration::from_secs(60),
            )
            .unwrap(),
        );
        let result = super::checked(
            super::process::output(&mut Command::new(binary), Duration::from_secs(5)).unwrap(),
        );
        result.lines().map(|line| line.parse().unwrap()).collect()
    }

    pub fn msl(&self, directory: &std::path::Path) -> Vec<u32> {
        let inputs = self
            .inputs
            .iter()
            .map(|v| format!("{v}u"))
            .collect::<Vec<_>>()
            .join(",");
        let result = super::execute_msl(
            directory,
            &format!(
                "uint input[] = {{{inputs}}}; uint output[257]; uint n=257; for (uint i=0; i<n; ++i) {{ seeded(input, output, n, {{i,0,0}}, {{0,0,0}}, {{n,1,1}}, {{1,1,1}}); std::cout << output[i] << '\\n'; }}"
            ),
        );
        result.lines().map(|line| line.parse().unwrap()).collect()
    }
}
