use std::{
    path::{Path, PathBuf},
    process::Command,
};

use crate::process::{self, Result};
use metal_oxide_artifact::{COMPILER_NIGHTLY, CompilerInfo};

fn source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal-oxide-compiler/Cargo.toml")
}

fn installed() -> Result<Option<PathBuf>> {
    if let Some(path) = std::env::var_os("METAL_OXIDE_COMPILER") {
        return Ok(Some(PathBuf::from(path).canonicalize()?));
    }
    let executable = std::env::current_exe()?.canonicalize()?;
    let adjacent = executable.with_file_name("metal-oxide-compiler");
    if adjacent.is_file() {
        return Ok(Some(adjacent));
    }
    if executable.parent()
        == Path::new(env!("METAL_OXIDE_DEV_BIN_DIR"))
            .canonicalize()
            .ok()
            .as_deref()
        && source().is_file()
    {
        return Ok(None);
    }
    Err("no installed metal-oxide-compiler; install the complete private bundle or set METAL_OXIDE_COMPILER".into())
}

pub(crate) fn resolve(directory: &Path, rustc: &str) -> Result<PathBuf> {
    if let Some(path) = installed()? {
        return Ok(path);
    }
    let host = rustc
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or("rustc did not report its host target")?;
    let messages = process::capture(
        Command::new("rustup")
            .current_dir(source().parent().unwrap())
            .env_remove("RUSTFLAGS")
            .env_remove("CARGO_ENCODED_RUSTFLAGS")
            .args([
                "run",
                COMPILER_NIGHTLY,
                "cargo",
                "build",
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
            .arg(source())
            .arg("--target-dir")
            .arg(directory),
    )?;
    let mut binaries = Vec::new();
    for line in messages.lines() {
        let message: serde_json::Value = serde_json::from_str(line)?;
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
    info.verify(rustc, &std::fs::read(path)?)?;
    Ok(info)
}

pub(crate) fn doctor() -> Result<String> {
    let Some(path) = installed()? else {
        return Ok(format!("development source: {}", source().display()));
    };
    let rustc =
        process::capture(Command::new("rustup").args(["run", COMPILER_NIGHTLY, "rustc", "-vV"]))?;
    let info = verify(&path, &rustc)?;
    Ok(format!(
        "{} ({}, ABI {})",
        path.display(),
        info.version,
        info.abi
    ))
}
