#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod run;

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
fn main() -> metal_oxide::Result<()> {
    run::verify()?;
    run::verify_chain()
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact)))]
fn main() -> std::process::ExitCode {
    eprintln!("run with cargo metal run -p particle-update on macOS Apple Silicon");
    std::process::ExitCode::FAILURE
}

#[cfg(all(
    test,
    target_os = "macos",
    target_arch = "aarch64",
    metal_oxide_artifact
))]
mod tests;
