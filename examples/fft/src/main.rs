#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod run;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> metal_oxide::Result<()> {
    run::verify()
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("fft runs on macOS with Apple Silicon");
}

#[cfg(test)]
mod tests;
