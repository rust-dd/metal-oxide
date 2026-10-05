use std::process::{Command, ExitCode};

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
    let result = metal_oxide::Device::system_default()
        .map(|device| device.name())
        .map_err(|error| error.to_string());

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let result = Err("GPU execution requires macOS on Apple Silicon".into());

    Check {
        name: "Metal device",
        result,
    }
}

pub(super) fn run() -> ExitCode {
    let checks = [
        probe("Rust", "rustc", &["--version"]),
        probe("Xcode", "xcodebuild", &["-version"]),
        probe(
            "macOS SDK",
            "xcrun",
            &["--sdk", "macosx", "--show-sdk-version"],
        ),
        probe(
            "Metal compiler",
            "xcrun",
            &["--sdk", "macosx", "metal", "--version"],
        ),
        gpu(),
    ];
    let mut healthy = true;
    for check in checks {
        match check.result {
            Ok(detail) => println!("[ok] {}: {}", check.name, detail.replace('\n', "; ")),
            Err(detail) => {
                healthy = false;
                println!("[missing] {}: {}", check.name, detail.replace('\n', "; "));
                if check.name == "Metal compiler" {
                    println!("  Install with: xcodebuild -downloadComponent MetalToolchain");
                    println!(
                        "  Runtime source compilation can still work; offline .metallib builds need this component."
                    );
                }
            }
        }
    }
    if healthy {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
