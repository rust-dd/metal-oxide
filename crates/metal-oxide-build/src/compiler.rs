use std::{
    path::{Path, PathBuf},
    process::Command,
};

use crate::{
    Error,
    metadata::Metadata,
    process::{self, Result},
};
use metal_oxide_artifact::{COMPILER_NIGHTLY, CompilerInfo, sha256};

/// The metal-oxide compiler that builds kernel crates.
pub enum Compiler {
    /// An installed `metal-oxide-compiler` executable.
    Executable(PathBuf),
    /// The `metal-oxide-compiler` package with this Cargo.toml, built with the pinned
    /// nightly into the kernels' Cargo target directory on first use.
    Source(PathBuf),
}

impl Compiler {
    /// `METAL_OXIDE_COMPILER`, then the compiler sources beside the `metal-oxide-device` that
    /// `metadata` resolves, then `metal-oxide-compiler` on `PATH`.
    pub(crate) fn search(metadata: &Metadata) -> Result<Self> {
        if let Some(path) = std::env::var_os("METAL_OXIDE_COMPILER") {
            return Ok(Self::Executable(PathBuf::from(path).canonicalize()?));
        }
        if let Some(source) = metadata.compiler_source() {
            return Ok(Self::Source(source));
        }
        let paths = std::env::var_os("PATH").unwrap_or_default();
        std::env::split_paths(&paths)
            .map(|directory| directory.join("metal-oxide-compiler"))
            .find(|path| path.is_file())
            .map(Self::Executable)
            .ok_or_else(|| format!("no metal-oxide-compiler found; {}", installation()).into())
    }

    /// The installed compiler's version and ABI after checking it, or the source it builds from.
    pub fn describe(&self) -> std::result::Result<String, Error> {
        match self {
            Self::Executable(path) => {
                let rustc = process::capture(Command::new("rustup").args([
                    "run",
                    COMPILER_NIGHTLY,
                    "rustc",
                    "-vV",
                ]))
                .map_err(Error)?;
                let info = verify(path, &rustc).map_err(Error)?;
                Ok(format!(
                    "{} ({}, ABI {})",
                    path.display(),
                    info.version,
                    info.abi
                ))
            }
            Self::Source(manifest) => Ok(format!("development source: {}", manifest.display())),
        }
    }

    pub(crate) fn executable(&self, directory: &Path, rustc: &str) -> Result<PathBuf> {
        match self {
            Self::Executable(path) => Ok(path.clone()),
            Self::Source(manifest) => build(manifest, directory, rustc),
        }
    }
}

/// Builds the compiler below `directory`, in a subdirectory per source location: Cargo
/// would otherwise reuse a binary built from another checkout of the same workspace.
fn build(manifest: &Path, directory: &Path, rustc: &str) -> Result<PathBuf> {
    let source = sha256(manifest.to_string_lossy().as_bytes());
    let directory = directory.join(&source[..16]);
    let host = rustc
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or("rustc did not report its host target")?;
    let messages = process::capture(
        process::isolate(&mut Command::new("rustup"))
            .current_dir(manifest.parent().unwrap())
            .args([
                "run",
                COMPILER_NIGHTLY,
                "cargo",
                "build",
                "--release",
                "--features",
                "rustc-private",
                "--bin",
                "metal-oxide-compiler",
                "--locked",
                "--message-format=json-render-diagnostics",
                "--target",
                host,
            ])
            .arg("--manifest-path")
            .arg(manifest)
            .arg("--target-dir")
            .arg(&directory),
    )?;
    let mut binaries = Vec::new();
    for line in messages.lines() {
        let message = serde_json::from_str::<serde_json::Value>(line)?;
        if message["reason"] == "compiler-artifact"
            && message["target"]["name"] == "metal-oxide-compiler"
            && let Some(path) = message["executable"].as_str()
        {
            binaries.push(PathBuf::from(path));
        }
    }
    match binaries.as_slice() {
        [path] => Ok(path.canonicalize()?),
        _ => Err("Cargo did not report exactly one metal-oxide-compiler executable".into()),
    }
}

pub(crate) fn verify(path: &Path, rustc: &str) -> Result<CompilerInfo> {
    let response = process::capture(Command::new("rustup").args(["run", COMPILER_NIGHTLY]).arg(path).arg("--metal-compiler-info"))
        .map_err(|error| format!("cannot run compiler {} with {COMPILER_NIGHTLY}: {error}; install matching rustc-dev and llvm-tools components", path.display()))?;
    let info = CompilerInfo::from_json(&response)?;
    info.verify(rustc, &std::fs::read(path)?)
        .map_err(|error| format!("{error} in {}; {}", path.display(), installation()))?;
    Ok(info)
}

fn installation() -> String {
    format!(
        "install the matching compiler with `cargo install cargo-metal --version '={}' --locked` and `cargo metal setup`, or set METAL_OXIDE_COMPILER",
        env!("CARGO_PKG_VERSION")
    )
}
