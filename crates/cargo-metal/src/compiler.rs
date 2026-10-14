use std::{
    path::{Path, PathBuf},
    process::Command,
};

use metal_oxide_artifact::COMPILER_NIGHTLY;
use metal_oxide_build::Compiler;

/// The compiler source of a development build of cargo-metal, which kernels in this
/// checkout build on first use.
fn development_source() -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?.canonicalize().ok()?;
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal-oxide-compiler/Cargo.toml");
    let development = Path::new(env!("METAL_OXIDE_DEV_BIN_DIR"))
        .canonicalize()
        .ok();
    (executable.parent() == development.as_deref() && source.is_file())
        .then(|| source.canonicalize().ok())
        .flatten()
}

/// `METAL_OXIDE_COMPILER`, then the compiler installed beside this executable, then the
/// source of a development build.
pub(crate) fn find() -> Result<Compiler, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(path) = std::env::var_os("METAL_OXIDE_COMPILER") {
        return Ok(Compiler::Executable(PathBuf::from(path).canonicalize()?));
    }
    let adjacent = std::env::current_exe()?
        .canonicalize()?
        .with_file_name("metal-oxide-compiler");
    if adjacent.is_file() {
        return Ok(Compiler::Executable(adjacent));
    }
    development_source()
        .map(Compiler::Source)
        .ok_or_else(|| "no metal-oxide-compiler beside cargo-metal; run cargo metal setup or set METAL_OXIDE_COMPILER".into())
}

/// Installs the compiler beside this executable, from the checkout cargo-metal was built from
/// or from crates.io; a development build and `METAL_OXIDE_COMPILER` need no installation.
pub(crate) fn install() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    if std::env::var_os("METAL_OXIDE_COMPILER").is_some() || development_source().is_some() {
        return Ok(());
    }
    let executable = std::env::current_exe()?.canonicalize()?;
    let root = executable
        .parent()
        .and_then(Path::parent)
        .ok_or("cargo-metal is not installed in a bin directory")?;
    let mut command = Command::new("rustup");
    command
        .args(["run", COMPILER_NIGHTLY, "cargo", "install", "--locked"])
        .args(["--features", "rustc-private", "--root"])
        .arg(root);
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal-oxide-compiler");
    if source.join("Cargo.toml").is_file() {
        command.arg("--path").arg(source);
    } else {
        command.args([
            "metal-oxide-compiler",
            "--version",
            concat!("=", env!("CARGO_PKG_VERSION")),
        ]);
    }
    let status = command.status()?;
    if !status.success() {
        return Err(format!("installing metal-oxide-compiler exited with {status}").into());
    }
    Ok(())
}
