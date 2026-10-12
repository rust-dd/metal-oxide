use std::path::{Path, PathBuf};

use metal_oxide_build::Compiler;

/// `METAL_OXIDE_COMPILER`, then the compiler installed beside this executable. A
/// development build of cargo-metal compiles the compiler from this repository instead.
pub(crate) fn find() -> Result<Compiler, Box<dyn std::error::Error + Send + Sync>> {
    if let Some(path) = std::env::var_os("METAL_OXIDE_COMPILER") {
        return Ok(Compiler::Executable(PathBuf::from(path).canonicalize()?));
    }
    let executable = std::env::current_exe()?.canonicalize()?;
    let adjacent = executable.with_file_name("metal-oxide-compiler");
    if adjacent.is_file() {
        return Ok(Compiler::Executable(adjacent));
    }
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal-oxide-compiler/Cargo.toml");
    if executable.parent()
        == Path::new(env!("METAL_OXIDE_DEV_BIN_DIR"))
            .canonicalize()
            .ok()
            .as_deref()
        && source.is_file()
    {
        return Ok(Compiler::Source(source.canonicalize()?));
    }
    Err("no installed metal-oxide-compiler; install the complete bundle or set METAL_OXIDE_COMPILER".into())
}
