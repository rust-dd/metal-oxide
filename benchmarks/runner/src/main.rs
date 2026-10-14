#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod gpu;
#[cfg(any(test, all(target_os = "macos", target_arch = "aarch64")))]
mod stats;

/// Runs the workloads of the configuration that `scripts/bench.py` writes.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    gpu::run()
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("benchmarks run on macOS with Apple Silicon");
}
