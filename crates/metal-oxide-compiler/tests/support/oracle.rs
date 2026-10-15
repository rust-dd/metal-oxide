use super::differential::{Case, Run};
use std::path::PathBuf;

/// Why a program fails the oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Failure {
    /// rustc rejects the program or its build does not finish: a generator bug.
    Invalid(String),
    /// The compiler rejects or crashes on a program that rustc accepts.
    Compiler(String),
    /// The emitted MSL does not run on the host, or its words differ from the CPU's.
    Msl(String),
    /// Metal rejects the emitted MSL, or the GPU differs from the CPU.
    Gpu(String),
}

impl Failure {
    /// Whether `other` fails at the same stage and, for the compiler, with the same first
    /// diagnostic, wherever it points.
    pub fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Compiler(a), Self::Compiler(b)) => diagnostic(a) == diagnostic(b),
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (stage, detail) = match self {
            Self::Invalid(detail) => ("rustc rejects or cannot run the program", detail),
            Self::Compiler(detail) => ("the compiler fails", detail),
            Self::Msl(detail) => ("the emitted MSL differs on the host", detail),
            Self::Gpu(detail) => ("the GPU differs", detail),
        };
        write!(f, "{stage}: {detail}")
    }
}

impl Case {
    /// The CPU outcomes and emitted kernels, after comparing the host-built MSL with the CPU;
    /// the host runs one thread at a time, so cooperative programs skip the MSL.
    pub fn check(&self) -> Result<(Run, PathBuf), Failure> {
        let expected = self.cpu().map_err(Failure::Invalid)?;
        let generated = self.compile().map_err(Failure::Compiler)?;
        if self.cooperative {
            return Ok((expected, generated));
        }
        let actual = self.msl(&generated).map_err(Failure::Msl)?;
        match self.mismatch(&expected.threads, &actual) {
            Some(mismatch) => Err(Failure::Msl(mismatch)),
            None => Ok((expected, generated)),
        }
    }

    /// The oracle on the GPU for the outcomes and kernels of [`Case::check`].
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub fn check_gpu(
        &self,
        device: &metal_oxide::Device,
        expected: &Run,
        generated: &std::path::Path,
    ) -> Result<(), Failure> {
        let gpu = self.gpu(device, generated).map_err(Failure::Gpu)?;
        match self.gpu_mismatch(expected, gpu.failure, &gpu.words, &gpu.counters) {
            Some(mismatch) => Err(Failure::Gpu(mismatch)),
            None => Ok(()),
        }
    }
}

/// The first diagnostic of a compiler failure without its location, which a smaller program
/// moves.
fn diagnostic(failure: &str) -> &str {
    let mut lines = failure.lines();
    while let Some(line) = lines.next() {
        if line.contains("panicked at") {
            return lines.next().map_or(line, str::trim);
        }
        if let Some(message) = line.strip_prefix("error: ") {
            return match message.find(".rs:") {
                Some(at) => message[at..]
                    .splitn(4, ':')
                    .nth(3)
                    .map_or(message, str::trim),
                None => message,
            };
        }
    }
    failure
}
