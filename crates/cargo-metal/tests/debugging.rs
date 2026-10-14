#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide_artifact::{ArtifactFile, Manifest, MathMode};
use support::{Project, checked};

const ADD: &str = r#"
use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};
use metal_oxide::{Device, LaunchConfig, Operation};

#[metal_oxide::kernel(requires(a.len() == out.len(), b.len() == out.len()))]
pub fn add(a: ReadBuffer<f32>, b: ReadBuffer<f32>, mut out: DisjointBuffer<f32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        let i = index.get();
        *out = a.as_slice()[i] + b.as_slice()[i];
    }
}

fn main() -> metal_oxide::Result<()> {
    let device = Device::system_default()?;
    let a = device.buffer_from_slice(&[1.0_f32, 2.0, 3.0])?;
    let b = device.buffer_from_slice(&[10.0_f32, 20.0, 30.0])?;
    let mut out = device.buffer_zeroed::<f32>(3)?;
    add(&device, LaunchConfig::<64>::for_elements(3)?, &a, &b, &mut out).wait()?;
    assert_eq!(out.read().wait(), [11.0, 22.0, 33.0]);
    Ok(())
}
"#;

fn names_source(artifact: &std::path::Path) -> bool {
    let air = std::fs::read(artifact.join(ArtifactFile::Air.name())).unwrap();
    air.windows(11).any(|name| name == b"src/main.rs")
}

fn metal_flags(artifact: &std::path::Path) -> Vec<String> {
    let manifest = std::fs::read_to_string(artifact.join(ArtifactFile::Manifest.name())).unwrap();
    Manifest::from_json(&manifest).unwrap().build.metal_flags
}

#[test]
#[ignore = "requires a Metal device, the compiler nightly, rust-src and the Metal toolchain"]
fn captures_record_gpu_work_with_rust_line_tables() {
    let project = Project::new(ADD);
    let directory = project.0.join("traces");
    std::fs::create_dir(&directory).unwrap();
    // A relative trace path starts where cargo metal runs.
    checked(
        project
            .metal(&["run", "--quiet", "--capture", "add.gputrace"])
            .current_dir(&directory)
            .output()
            .unwrap(),
    );
    let trace = directory.join("add.gputrace");
    assert!(trace.join("metadata").is_file());
    let recorded = std::fs::read(trace.join("capture")).unwrap();
    assert!(recorded.windows(3).any(|name| name == b"add"));
    let captured = project.artifact_of("add", MathMode::Precise, true);
    let root = format!("-fdebug-compilation-dir={}", project.0.display());
    for flag in ["-gline-tables-only", "-frecord-sources", root.as_str()] {
        assert!(metal_flags(&captured).iter().any(|f| f == flag), "{flag}");
    }
    assert!(names_source(&captured));

    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let plain = project.artifact("add");
    assert_ne!(plain, captured);
    assert!(
        !metal_flags(&plain)
            .iter()
            .any(|flag| flag.starts_with("-g"))
    );
    assert!(!names_source(&plain));
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn inspect_prints_the_kernels_a_build_compiled() {
    let project = Project::new(ADD);
    for (emit, expected) in [
        ("ir", "add"),
        ("msl", "kernel void add"),
        ("air", "define "),
    ] {
        let output = checked(
            project
                .metal(&["inspect", "--emit", emit])
                .output()
                .unwrap(),
        );
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(text.starts_with("// host::add: add\n"), "{text}");
        assert!(text.contains(expected), "{emit}: {text}");
    }
    let output = project
        .metal(&["inspect", "--emit", "msl", "--kernel", "missing"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no kernel named missing"));
}

#[test]
#[ignore = "requires a Metal device, the compiler nightly, rust-src and the Metal toolchain"]
fn validation_runs_the_host_under_metal_validation() {
    let project = Project::new(ADD);
    checked(
        project
            .metal(&["run", "--quiet", "--validate"])
            .output()
            .unwrap(),
    );
}
