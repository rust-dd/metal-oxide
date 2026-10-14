#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide_artifact::{ArtifactFile, Manifest};
use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
};
use support::{Workspace, checked};

const KERNELS: &str = include_str!("../../../examples/vec-add/kernels/src/lib.rs");
const HOST: &str = r#"mod kernels {
    include!(env!("METAL_OXIDE_BINDINGS"));
}

fn main() -> metal_oxide::Result<()> {
    use metal_oxide::{Device, LaunchConfig, Operation};
    let device = Device::system_default()?;
    let kernels = kernels::load(&device, env!("METAL_OXIDE_ARTIFACT_DIR"))?;
    let a = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0])?;
    let b = device.buffer_from_slice(&[10.0_f32, 20.0, 30.0])?;
    let mut out = device.buffer_zeroed::<f32>(3)?;
    kernels.vec_add(LaunchConfig::<64>::for_elements(3)?, &a, &b, &mut out).wait()?;
    assert_eq!(out.read().wait(), [11.0, 22.0, 33.0]);
    Ok(())
}
"#;

/// The directory and manifest of the artifact that a cargo-metal command built.
fn built(output: Output) -> (PathBuf, Manifest) {
    let log = String::from_utf8(checked(output).stderr).unwrap();
    let directory = log
        .lines()
        .find_map(|line| line.strip_prefix("Built Metal artifact "))
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{log}"));
    let manifest = std::fs::read_to_string(directory.join(ArtifactFile::Manifest.name())).unwrap();
    (directory, Manifest::from_json(&manifest).unwrap())
}

fn names_kernel_source(directory: &Path) -> bool {
    let air = std::fs::read(directory.join(ArtifactFile::Air.name())).unwrap();
    air.windows(18).any(|name| name == b"kernels/src/lib.rs")
}

#[test]
#[ignore = "requires a Metal device, compiler nightly, rust-src and Metal toolchain"]
fn captures_record_gpu_work_with_rust_line_tables() {
    let workspace = Workspace::new();
    let runtime = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal-oxide")
        .canonicalize()
        .unwrap();
    let root = workspace.0.join("Cargo.toml");
    let members = std::fs::read_to_string(&root).unwrap();
    std::fs::write(
        &root,
        format!("{members}[workspace.dependencies]\nmetal-oxide={{path={runtime:?}}}\n"),
    )
    .unwrap();
    let host = workspace.0.join("host/Cargo.toml");
    let package = std::fs::read_to_string(&host).unwrap();
    std::fs::write(
        &host,
        format!("{package}[dependencies]\nmetal-oxide={{workspace=true}}\n"),
    )
    .unwrap();
    std::fs::write(workspace.0.join("host/src/main.rs"), HOST).unwrap();
    std::fs::write(workspace.0.join("kernels/src/lib.rs"), KERNELS).unwrap();
    checked(
        Command::new("cargo")
            .current_dir(&workspace.0)
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );

    // The host runs in the workspace root; a relative trace path starts where cargo-metal runs.
    let (captured, capture) = built(
        workspace
            .command("run")
            .current_dir(workspace.0.join("host"))
            .args(["--capture", "traces/vec_add.gputrace"])
            .output()
            .unwrap(),
    );
    let trace = workspace.0.join("host/traces/vec_add.gputrace");
    assert!(trace.join("metadata").is_file());
    let recorded = std::fs::read(trace.join("capture")).unwrap();
    assert!(recorded.windows(7).any(|name| name == b"vec_add"));

    let (plain, manifest) = built(workspace.command("build").output().unwrap());
    assert_ne!(plain, captured);
    let directory = format!("-fdebug-compilation-dir={}", workspace.0.display());
    let debug = ["-gline-tables-only", "-frecord-sources", &directory].map(str::to_owned);
    assert_eq!(
        capture.build.metal_flags,
        [manifest.build.metal_flags.as_slice(), &debug].concat()
    );
    assert!(names_kernel_source(&captured));
    assert!(!names_kernel_source(&plain));
}
