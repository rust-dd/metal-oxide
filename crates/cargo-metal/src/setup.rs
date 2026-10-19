use std::process::{Command, ExitCode};

use metal_oxide_artifact::COMPILER_NIGHTLY as NIGHTLY;

struct Check {
    name: &'static str,
    result: Result<String, String>,
}

fn probe(name: &'static str, program: &str, arguments: &[&str]) -> Check {
    let result = Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| error.to_string())
        .and_then(|output| {
            if output.status.success() {
                Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
            } else {
                let message = String::from_utf8_lossy(&output.stderr).trim().to_owned();
                Err(if message.is_empty() {
                    output.status.to_string()
                } else {
                    message
                })
            }
        });
    Check { name, result }
}

fn gpu() -> Check {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let result = limits().map_err(|error| error.to_string());

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let result = Err("GPU execution requires macOS on Apple Silicon".into());

    Check {
        name: "Metal device",
        result,
    }
}

/// The device limits that kernels and launch planning rely on.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn limits() -> metal_oxide::Result<String> {
    use metal_oxide::{Device, Module, Pipeline};
    let device = Device::system_default()?;
    let module = Module::from_source(
        &device,
        "kernel void probe(device uint* out [[buffer(0)]]) { out[0] = 1u; }",
    )?;
    let pipeline = Pipeline::new(&device, &module, "probe")?;
    Ok(format!(
        "{}; SIMD width {}, {} threads per block, {} B threadgroup memory",
        device.name(),
        pipeline.thread_execution_width(),
        pipeline.max_threads_per_block(),
        device.max_threadgroup_memory_per_block()
    ))
}

/// Installs the pinned nightly with its components, Apple's Metal Toolchain when it is missing
/// and the metal-oxide compiler, unless only `check`ing; then reports the environment.
pub(super) fn run(check: bool) -> ExitCode {
    if !check && let Err(error) = install() {
        eprintln!("cargo-metal: {error}");
        return ExitCode::FAILURE;
    }
    report()
}

fn install() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let status = Command::new("rustup")
        .args([
            "toolchain",
            "install",
            NIGHTLY,
            "--profile",
            "minimal",
            "--no-self-update",
        ])
        .args(["--component", "rustc-dev", "--component", "rust-src"])
        .args(["--component", "llvm-tools"])
        .status()?;
    if !status.success() {
        return Err(format!("installing {NIGHTLY} exited with {status}").into());
    }
    if metal_oxide_build::metal_compiler().is_err() {
        let status = Command::new("xcodebuild")
            .args(["-downloadComponent", "MetalToolchain"])
            .status()?;
        if !status.success() {
            return Err(format!("downloading the Metal Toolchain exited with {status}").into());
        }
    }
    crate::compiler::install()
}

fn report() -> ExitCode {
    let checks = [
        probe("macOS", "sw_vers", &["-productVersion"]),
        probe("Rust", "rustc", &["--version"]),
        probe(
            "Compiler Rust",
            "rustup",
            &["run", NIGHTLY, "rustc", "--version"],
        ),
        components(),
        Check {
            name: "metal-oxide compiler",
            result: crate::compiler::find()
                .map_err(|e| e.to_string())
                .and_then(|compiler| compiler.describe().map_err(|e| e.to_string())),
        },
        probe("Xcode", "xcodebuild", &["-version"]),
        probe(
            "macOS SDK",
            "xcrun",
            &["--sdk", "macosx", "--show-sdk-version"],
        ),
        Check {
            name: "Metal compiler",
            result: metal_oxide_build::metal_compiler().map_err(|e| e.to_string()),
        },
        gpu(),
    ];
    let mut healthy = true;
    for check in checks {
        match check.result {
            Ok(detail) => println!("[ok] {}: {}", check.name, detail.replace('\n', "; ")),
            Err(detail) => {
                healthy = false;
                println!("[missing] {}: {}", check.name, detail.replace('\n', "; "));
            }
        }
    }
    if healthy {
        ExitCode::SUCCESS
    } else {
        println!("Install what is missing with: cargo metal setup");
        ExitCode::FAILURE
    }
}

fn components() -> Check {
    let result = probe(
        "Compiler components",
        "rustup",
        &["component", "list", "--installed", "--toolchain", NIGHTLY],
    )
    .result
    .and_then(|installed| {
        for name in ["rustc-dev", "rust-src", "llvm-tools"] {
            if !installed
                .lines()
                .any(|line| line == name || line.starts_with(&format!("{name}-")))
            {
                return Err(format!("{NIGHTLY} is missing {name}"));
            }
        }
        Ok("rustc-dev, rust-src, llvm-tools".into())
    });
    Check {
        name: "Compiler components",
        result,
    }
}
