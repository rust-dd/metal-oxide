#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod program;
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod verify;
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod workloads;

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    verify::all()?;
    Ok(())
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact)))]
fn main() -> std::process::ExitCode {
    eprintln!("run with cargo metal run -p cooperation on macOS Apple Silicon");
    std::process::ExitCode::FAILURE
}
#[cfg(all(
    test,
    target_os = "macos",
    target_arch = "aarch64",
    metal_oxide_artifact
))]
mod tests;
