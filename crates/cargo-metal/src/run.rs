use crate::cli::Cargo;
use std::process::{Command, ExitCode};

/// Cargo as `cargo metal` was invoked through, or the one on `PATH`.
pub(crate) fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

/// Runs `cargo run` or `cargo test` with the environment that Metal debugging needs; the
/// kernel macros build capture runs with line tables.
pub(crate) fn execute(
    options: &Cargo,
) -> Result<ExitCode, Box<dyn std::error::Error + Send + Sync>> {
    let mut command = cargo();
    command
        .arg(if options.test { "test" } else { "run" })
        .args(&options.arguments);
    if options.validate {
        command
            .env("MTL_DEBUG_LAYER", "1")
            .env("MTL_SHADER_VALIDATION", "1");
    }
    if let Some(trace) = &options.capture {
        command
            .env("METAL_CAPTURE_ENABLED", "1")
            .env("METAL_OXIDE_CAPTURE", std::path::absolute(trace)?);
    }
    let status = command.status()?;
    Ok(ExitCode::from(status.code().map_or(1, |code| code as u8)))
}
