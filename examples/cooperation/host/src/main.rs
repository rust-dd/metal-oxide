#[metal_oxide::kernels(path = "../kernels")]
mod kernels {}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod program;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod verify;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod workloads;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    verify::all()?;
    Ok(())
}
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("cooperation runs on macOS with Apple Silicon");
}
#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests;
