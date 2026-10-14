#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod run;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> metal_oxide::Result<()> {
    run::verify()?;
    run::verify_chain()
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("particle-update runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests;
