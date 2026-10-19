use std::{fmt, process::Command};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub(crate) const ISOLATED_ENV: [&str; 12] = [
    "RUSTC",
    "RUSTDOC",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTC_LINKER",
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "RUSTDOCFLAGS",
    "CARGO_ENCODED_RUSTDOCFLAGS",
    "CARGO_BUILD_TARGET",
    "CARGO_TARGET_DIR",
    "RUSTUP_TOOLCHAIN",
];

/// A kernel build or artifact check that failed, with the failing step's message.
pub struct Error(pub(crate) Box<dyn std::error::Error + Send + Sync>);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

// Debug prints the message alone, so an unwrapped failure reads like its Display form.
impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for Error {}

pub(crate) fn capture(command: &mut Command) -> Result<String> {
    let output = command.output()?;
    if !output.status.success() {
        let program = command.get_program().to_string_lossy();
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(if stderr.trim().is_empty() {
            format!("{program} exited with {}", output.status)
        } else {
            format!("{program}: {}", stderr.trim())
        }
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

pub(crate) fn run(command: &mut Command) -> Result<()> {
    let status = command.status()?;
    if !status.success() {
        return Err(format!(
            "{} exited with {status}",
            command.get_program().to_string_lossy()
        )
        .into());
    }
    Ok(())
}

/// Drops the host build's toolchain, wrapper, flag and target settings so nested builds use
/// their own.
pub(crate) fn isolate(command: &mut Command) -> &mut Command {
    for name in ISOLATED_ENV {
        command.env_remove(name);
    }
    command
}
