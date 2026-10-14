use super::generate::{Program, Word};
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

/// The seeds that every run of the compiler tests checks; fuzzing draws others.
pub const SEEDS: [u64; 16] = [
    0,
    1,
    2,
    3,
    7,
    31,
    255,
    257,
    1023,
    65535,
    0x1234_5678,
    0xdead_beef,
    0x8000_0000,
    u32::MAX as u64,
    0x0123_4567_89ab_cdef,
    u64::MAX,
];

const THREADS: usize = 257;

/// A generated program, its inputs and the directory that holds its outputs.
pub struct Case {
    pub directory: PathBuf,
    pub source: PathBuf,
    pub program: Program,
    pub inputs: Vec<u32>,
}

impl Case {
    pub fn generated(seed: u64) -> Self {
        let directory = scratch(&format!("{seed:016x}"));
        let mut state = seed as u32 ^ (seed >> 32) as u32;
        let mut inputs = vec![0, 1, u32::MAX, 0x7fff_ffff, 0x8000_0000, 65535, 65536];
        while inputs.len() < THREADS {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            inputs.push(state);
        }
        let program = Program::generate(seed);
        let source = directory.join("case.rs");
        std::fs::write(&source, program.source(&inputs)).unwrap();
        Self {
            directory,
            source,
            program,
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

    /// The words of every thread, from rustc's build of the program for the host.
    pub fn cpu(&self) -> Vec<Vec<u32>> {
        let binary = self.directory.join("cpu");
        let mut rustc = Command::new("rustc");
        rustc
            .arg(&self.source)
            .args(["--edition=2024", "--cfg", "cpu_reference", "-O"])
            .args(["-C", "overflow-checks=off", "-o"])
            .arg(&binary);
        super::checked(super::process::output(&mut rustc, Duration::from_secs(120)).unwrap());
        let result = super::checked(
            super::process::output(&mut Command::new(binary), Duration::from_secs(10)).unwrap(),
        );
        words(&result)
    }

    /// The words of every thread, from the emitted MSL compiled for the host.
    pub fn msl(&self, directory: &Path) -> Vec<Vec<u32>> {
        let words = self.program.words.len();
        let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
        let inputs = self
            .inputs
            .iter()
            .map(|v| format!("{v}u"))
            .collect::<Vec<_>>();
        let main = format!(
            "uint input[] = {{{}}}; uint output[{}] = {{}}; uint n = {THREADS}; ulong lengths[] = {{{THREADS}, {}, 0}}; \
             for (uint i = 0; i < n; ++i) {{ generated({}); }} \
             for (uint i = 0; i < n * {words}; ++i) {{ std::cout << output[i] << ((i + 1) % {words} == 0 ? '\\n' : ' '); }}",
            inputs.join(","),
            THREADS * words,
            THREADS * words,
            invocation(&msl)
        );
        self::words(&super::execute_msl(directory, &main))
    }

    /// The first word in which `actual` differs from `expected`.
    pub fn mismatch(&self, expected: &[Vec<u32>], actual: &[Vec<u32>]) -> Option<String> {
        if expected.len() != actual.len() {
            return Some(format!(
                "{} threads, expected {}",
                actual.len(),
                expected.len()
            ));
        }
        for (thread, (expected, actual)) in expected.iter().zip(actual).enumerate() {
            for (index, ((&expected, &actual), kind)) in expected
                .iter()
                .zip(actual)
                .zip(&self.program.words)
                .enumerate()
            {
                if !agree(*kind, expected, actual) {
                    return Some(format!(
                        "thread {thread} (input {:#x}), word {index}: expected {expected:#010x}, got {actual:#010x}",
                        self.inputs[thread]
                    ));
                }
            }
        }
        None
    }
}

fn agree(kind: Word, expected: u32, actual: u32) -> bool {
    if expected == actual {
        return true;
    }
    let (expected, actual) = (f32::from_bits(expected), f32::from_bits(actual));
    kind == Word::Float
        && (expected.is_nan() && actual.is_nan() || expected.is_subnormal() && actual == 0.0)
}

fn words(text: &str) -> Vec<Vec<u32>> {
    text.lines()
        .map(|line| line.split(' ').map(|word| word.parse().unwrap()).collect())
        .collect()
}

/// The arguments of the emitted `generated` entry, by the names of its parameters.
fn invocation(msl: &str) -> String {
    let signature = msl
        .lines()
        .find(|line| line.starts_with("kernel void generated("))
        .expect("the emitted MSL has the generated kernel");
    let parameters = &signature[signature.find('(').unwrap() + 1..signature.rfind(')').unwrap()];
    let arguments = parameters.split(", ").map(|parameter| {
        let declaration = parameter.split(" [[").next().unwrap();
        match declaration.rsplit(' ').next().unwrap() {
            "metal_oxide_arg_1" => "input",
            "metal_oxide_arg_2" => "output",
            "metal_oxide_arg_3" => "n",
            "metal_oxide_thread_idx" => "{i % 64, 0, 0}",
            "metal_oxide_block_idx" => "{i / 64, 0, 0}",
            "metal_oxide_block_dim" => "{64, 1, 1}",
            "metal_oxide_grid_dim" => "{(n + 63) / 64, 1, 1}",
            "metal_oxide_thread_position" => "{i, 0, 0}",
            "metal_oxide_gate" => "metal_oxide_gate",
            "metal_oxide_failure" => "metal_oxide_failure",
            "metal_oxide_lengths" => "lengths",
            other => panic!("unexpected kernel parameter {other} in {signature}"),
        }
    });
    arguments.collect::<Vec<_>>().join(", ")
}

/// A directory of this process for sources that a test writes.
pub fn scratch(name: &str) -> PathBuf {
    let directory =
        super::root().join(format!("target/differential/{}-{name}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    directory
}

#[cfg(test)]
mod tests {
    use super::{Word, agree};

    #[test]
    fn floats_agree_on_nan_and_flushed_subnormals_and_integers_exactly() {
        let nan = f32::NAN.to_bits();
        assert!(agree(Word::Float, nan, (-f32::NAN).to_bits()));
        assert!(agree(Word::Float, f32::MIN_POSITIVE.to_bits() / 2, 0));
        assert!(!agree(
            Word::Float,
            1.0_f32.to_bits(),
            1.0000001_f32.to_bits()
        ));
        assert!(!agree(Word::Exact, nan, (-f32::NAN).to_bits()));
        assert!(agree(Word::Exact, 7, 7));
    }
}
