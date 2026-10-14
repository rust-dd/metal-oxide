#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod cpu;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod euler;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod fgn;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod native;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod sheet;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod validation;

#[cfg_attr(
    feature = "fast",
    metal_oxide::kernels(path = "../kernels", math = "fast")
)]
#[cfg_attr(not(feature = "fast"), metal_oxide::kernels(path = "../kernels"))]
mod kernels {}

/// The numerical mode of the kernels, which the native references follow.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const MATH_MODE: metal_oxide_artifact::MathMode = if cfg!(feature = "fast") {
    metal_oxide_artifact::MathMode::Fast
} else {
    metal_oxide_artifact::MathMode::Precise
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.is_empty() {
        validation::run()
    } else if arguments[0] == "--bench" && (2..=5).contains(&arguments.len()) {
        bench::run(
            std::path::Path::new(&arguments[1]),
            arguments
                .get(2)
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(20),
            arguments
                .get(3)
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(5),
            arguments
                .get(4)
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(16384),
        )
    } else {
        Err("usage: spectral [--bench OUTPUT [SAMPLES [WARMUP [MAX_ROWS]]]]".into())
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn main() {
    eprintln!("spectral runs on macOS with Apple Silicon");
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires a Metal device"]
fn generated_spectral_kernels_match_cpu_and_native() {
    validation::run().unwrap();
}
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
mod bench;
