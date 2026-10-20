use super::generate::{Program, Word};
use metal_oxide_artifact::{ArtifactFile, Check};
use std::{
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

/// The seeds that every run of the compiler tests checks; fuzzing draws others.
pub const SEEDS: [u64; 17] = [
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
    0x2c83_c990_4457_4110,
    u64::MAX,
];

const THREADS: usize = 257;

/// What one thread produced: its words, or the location of the panic or failed check that
/// stopped it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Words(Vec<u32>),
    Panic { line: u32, column: u32 },
}

/// What a program computed: an outcome for each thread with an input, and the final words of its
/// device counters.
#[derive(Clone, Debug, Default)]
pub struct Run {
    pub threads: Vec<Outcome>,
    pub counters: Vec<u32>,
}

/// What a GPU run produced: the check that failed, if one did, the words that the threads wrote
/// and the final words of the device counters.
pub struct Gpu {
    pub failure: Option<Outcome>,
    pub words: Vec<u32>,
    pub counters: Vec<u32>,
}

/// A program, its inputs and the directory that holds its outputs.
pub struct Case {
    pub directory: PathBuf,
    pub source: PathBuf,
    pub words: Vec<Word>,
    pub overflow_checks: bool,
    /// Whether the threads of a block first run `cooperate` together.
    pub cooperative: bool,
    /// The element type of the device counters that `cooperate` counts in, if it does.
    pub counters: Option<String>,
    pub inputs: Vec<u32>,
}

impl Case {
    pub fn generated(seed: u64) -> Self {
        Self::new(&format!("{seed:016x}"), &Program::generate(seed), seed)
    }

    /// `program` with the thread inputs of `seed`, written under a directory named `name`.
    pub fn new(name: &str, program: &Program, seed: u64) -> Self {
        let directory = scratch(name);
        let inputs = inputs(seed);
        let source = directory.join("case.rs");
        std::fs::write(&source, program.source(&inputs)).unwrap();
        Self {
            directory,
            source,
            words: program.words.clone(),
            overflow_checks: program.overflow_checks,
            cooperative: program.cooperative(),
            counters: program.counters(),
            inputs,
        }
    }

    /// A program that [`Program::source`] wrote, such as a minimized fixture; its header and CPU
    /// entry give the words, overflow checks, cooperation and inputs.
    pub fn saved(path: &Path) -> Self {
        let text = std::fs::read_to_string(path).unwrap();
        let field = |prefix: &str| {
            text.lines()
                .find_map(|line| line.strip_prefix(prefix))
                .unwrap_or_else(|| panic!("{} has no `{prefix}` line", path.display()))
                .to_owned()
        };
        let words = field("// words: ")
            .split(' ')
            .map(|word| match word {
                "exact" => Word::Exact,
                "float" => Word::Float,
                other => panic!("{}: unknown word {other}", path.display()),
            })
            .collect();
        let inputs = field("    let inputs = [")
            .trim_end_matches("];")
            .split(", ")
            .map(|input| input.trim_end_matches("_u32").parse().unwrap())
            .collect();
        Self {
            directory: scratch(path.file_stem().unwrap().to_str().unwrap()),
            source: path.to_path_buf(),
            words,
            overflow_checks: field("// overflow checks: ") == "on",
            cooperative: text.lines().any(|line| line == "// cooperative"),
            counters: text
                .lines()
                .find_map(|line| line.strip_prefix("// counters: "))
                .map(str::to_owned),
            inputs,
        }
    }

    fn overflow_checks(&self) -> &'static str {
        if self.overflow_checks {
            "overflow-checks=on"
        } else {
            "overflow-checks=off"
        }
    }

    /// The directory of the emitted kernels, or the compiler's output when it fails.
    pub fn compile(&self) -> Result<PathBuf, String> {
        let output = self.directory.join("generated");
        super::succeeded(super::emit_into(
            self.source.to_str().unwrap(),
            &["-C", self.overflow_checks()],
            &output,
        ))?;
        Ok(output)
    }

    /// What rustc's build of the program for the host computes.
    pub fn cpu(&self) -> Result<Run, String> {
        let binary = self.directory.join("cpu");
        let mut rustc = Command::new("rustc");
        rustc
            .arg(&self.source)
            .args(["--edition=2024", "--cfg", "cpu_reference", "-O", "-C"])
            .arg(self.overflow_checks())
            .arg("-o")
            .arg(&binary);
        let build = super::process::output(&mut rustc, Duration::from_secs(120))
            .map_err(|error| format!("rustc: {error}"))?;
        super::succeeded(build)?;
        let run =
            super::process::output(&mut Command::new(binary), super::process::EXECUTION_TIMEOUT)
                .map_err(|error| format!("the CPU build: {error}"))?;
        let mut computed = Run::default();
        for line in super::succeeded(run)?.lines() {
            if let Some(counters) = line.strip_prefix("counters ") {
                computed.counters = words(counters);
            } else if let Some(location) = line.strip_prefix("panic ") {
                let (line, column) = location.split_once(':').unwrap();
                computed.threads.push(Outcome::Panic {
                    line: line.parse().unwrap(),
                    column: column.parse().unwrap(),
                });
            } else {
                computed.threads.push(Outcome::Words(words(line)));
            }
        }
        Ok(computed)
    }

    /// What every thread produces when the emitted MSL runs on the host, one thread at a time.
    pub fn msl(&self, directory: &Path) -> Result<Vec<Outcome>, String> {
        let words = self.words.len();
        let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
        let inputs = self
            .inputs
            .iter()
            .map(|v| format!("{v}u"))
            .collect::<Vec<_>>();
        let main = format!(
            "uint input[] = {{{}}}; uint output[{}] = {{}}; uint n = {THREADS}; ulong lengths[] = {{{THREADS}, {}, 0}}; \
             for (uint i = 0; i < n; ++i) {{ \
                 generated({}); \
                 uint failure = atomic_load_explicit(metal_oxide_failure, memory_order_relaxed); \
                 atomic_store_explicit(metal_oxide_failure, 0u, memory_order_relaxed); \
                 if (failure != 0u) {{ std::cout << \"check \" << failure - 2u << '\\n'; continue; }} \
                 for (uint k = 0; k < {words}; ++k) {{ std::cout << output[i * {words} + k] << (k + 1 == {words} ? '\\n' : ' '); }} \
             }}",
            inputs.join(","),
            THREADS * words,
            THREADS * words,
            invocation(&msl)
        );
        let checks = checks(directory);
        Ok(super::run_msl(directory, &main)?
            .lines()
            .map(|line| match line.strip_prefix("check ") {
                Some(check) => checks[check.parse::<usize>().unwrap()].clone(),
                None => Outcome::Words(self::words(line)),
            })
            .collect())
    }

    /// The first thread whose outcome in `actual` differs from `expected`.
    pub fn mismatch(&self, expected: &[Outcome], actual: &[Outcome]) -> Option<String> {
        if expected.len() != actual.len() {
            return Some(format!(
                "{} threads, expected {}",
                actual.len(),
                expected.len()
            ));
        }
        expected
            .iter()
            .zip(actual)
            .position(|outcomes| match outcomes {
                (Outcome::Words(expected), Outcome::Words(actual)) => {
                    !self.words_agree(expected, actual)
                }
                (expected, actual) => expected != actual,
            })
            .map(|thread| {
                format!(
                    "thread {thread} (input {:#x}): expected {:x?}, got {:x?}",
                    self.inputs[thread], expected[thread], actual[thread]
                )
            })
    }

    /// Compares a GPU run with the CPU's: the one failed check must be a CPU panic, completed
    /// threads must have written their words, and the counters must end the same.
    pub fn gpu_mismatch(
        &self,
        expected: &Run,
        failure: Option<Outcome>,
        words: &[u32],
        counters: &[u32],
    ) -> Option<String> {
        if counters != expected.counters {
            let expected = &expected.counters;
            return Some(format!("counters {counters:x?}, expected {expected:x?}"));
        }
        let panics = expected
            .threads
            .iter()
            .filter(|outcome| matches!(outcome, Outcome::Panic { .. }))
            .collect::<Vec<_>>();
        match &failure {
            None if !panics.is_empty() => {
                return Some(format!(
                    "the GPU completed, the CPU panicked at {:?}",
                    panics[0]
                ));
            }
            Some(failure) if !panics.contains(&failure) => {
                return Some(format!(
                    "the GPU failed {failure:?}, which no CPU thread panicked at"
                ));
            }
            _ => {}
        }
        let actual = expected
            .threads
            .iter()
            .zip(words.chunks(self.words.len()))
            .map(|(expected, words)| match expected {
                Outcome::Words(_) => Outcome::Words(words.to_vec()),
                Outcome::Panic { .. } => expected.clone(),
            })
            .collect::<Vec<_>>();
        self.mismatch(&expected.threads, &actual)
    }

    fn words_agree(&self, expected: &[u32], actual: &[u32]) -> bool {
        expected.len() == actual.len()
            && expected
                .iter()
                .zip(actual)
                .zip(&self.words)
                .all(|((&expected, &actual), kind)| agree(*kind, expected, actual))
    }
}

/// The thread inputs of `seed`: edge values, then a linear congruential sequence.
fn inputs(seed: u64) -> Vec<u32> {
    let mut state = seed as u32 ^ (seed >> 32) as u32;
    let mut inputs = vec![0, 1, u32::MAX, 0x7fff_ffff, 0x8000_0000, 65535, 65536];
    while inputs.len() < THREADS {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        inputs.push(state);
    }
    inputs
}

/// The source location of each runtime check of a compiled case, by check index.
fn checks(directory: &Path) -> Vec<Outcome> {
    let checks = std::fs::read_to_string(directory.join(ArtifactFile::Checks.name())).unwrap();
    serde_json::from_str::<Vec<Check>>(&checks)
        .unwrap()
        .into_iter()
        .map(|check| Outcome::Panic {
            line: check.location.line,
            column: check.location.column,
        })
        .collect()
}

fn agree(kind: Word, expected: u32, actual: u32) -> bool {
    if expected == actual {
        return true;
    }
    let (expected, actual) = (f32::from_bits(expected), f32::from_bits(actual));
    kind == Word::Float
        && (expected.is_nan() && actual.is_nan() || expected.is_subnormal() && actual == 0.0)
}

fn words(line: &str) -> Vec<u32> {
    line.split(' ').map(|word| word.parse().unwrap()).collect()
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

/// The minimized programs under `tests/fixtures/generated/`, in name order.
pub fn fixtures() -> Vec<Case> {
    let directory = super::root().join("crates/metal-oxide-compiler/tests/fixtures/generated");
    let mut paths = std::fs::read_dir(directory)
        .map(|entries| {
            entries
                .map(|entry| entry.unwrap().path())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    paths.sort();
    paths.iter().map(|path| Case::saved(path)).collect()
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
