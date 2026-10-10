#[cfg(any(test, metal_oxide_artifact))]
mod cpu;
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod euler;
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod fgn;
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod native;
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod sheet;
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod validation;

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
#[allow(dead_code)]
mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
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

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact)))]
fn main() -> std::process::ExitCode {
    eprintln!("run with cargo metal run -p spectral on macOS Apple Silicon");
    std::process::ExitCode::FAILURE
}

#[cfg(all(
    test,
    target_os = "macos",
    target_arch = "aarch64",
    metal_oxide_artifact
))]
#[test]
#[ignore = "requires a Metal device and generated artifact"]
fn generated_spectral_kernels_match_cpu_and_native() {
    validation::run().unwrap();
}
#[cfg(all(target_os = "macos", target_arch = "aarch64", metal_oxide_artifact))]
mod bench;
