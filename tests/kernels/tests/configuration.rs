#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide_artifact::Manifest;
use std::path::PathBuf;
use support::{Project, checked};

const HOST: &str = "#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\nfn main() {}\n";
const KERNEL: &str = r#"#![no_std]
use metal_oxide_device::{DisjointBuffer, kernel, thread_index};
#[kernel]
pub fn add(mut out: DisjointBuffer<u32>, value: u32) {
    if let Some(out) = out.get_mut(&thread_index()) {
        *out = value + 1;
    }
}
"#;
const CREDENTIAL: &str = "configuration-credential-marker-should-not-reach-the-consumer-binary";

fn build(project: &Project) -> (PathBuf, Manifest) {
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let manifest = Manifest::from_json(
        &std::fs::read_to_string(
            project
                .0
                .join("metal/src/main.rs/kernels-precise/manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    (
        Project::target()
            .join("metal")
            .join(&manifest.build.fingerprint),
        manifest,
    )
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn cargo_configuration_changes_invalidate_cached_kernels() {
    let project = Project::new(HOST);
    project.kernel_crate(KERNEL);
    let configuration = project.0.join(".cargo/config.toml");
    std::fs::create_dir(configuration.parent().unwrap()).unwrap();
    std::fs::write(
        &configuration,
        format!("# {CREDENTIAL}\n[profile.release]\noverflow-checks = false\n"),
    )
    .unwrap();
    let (wrapping, manifest) = build(&project);
    assert!(manifest.checks.is_empty(), "{:?}", manifest.checks);
    let binary = std::fs::read(Project::target().join("debug/host")).unwrap();
    assert!(
        !binary
            .windows(CREDENTIAL.len())
            .any(|bytes| bytes == CREDENTIAL.as_bytes())
    );
    let inputs = std::fs::read_to_string(
        project
            .0
            .join("metal/src/main.rs/kernels-precise/inputs.json"),
    )
    .unwrap();
    assert!(!inputs.contains(CREDENTIAL));
    assert!(!inputs.contains(support::root().to_str().unwrap()));
    assert!(!inputs.contains(&std::env::var("HOME").unwrap()));
    std::fs::write(
        &configuration,
        "[profile.release]\noverflow-checks = true\n",
    )
    .unwrap();
    let (enabled, manifest) = build(&project);
    assert_ne!(enabled, wrapping, "configuration kept wrapping arithmetic");
    assert_eq!(manifest.checks.len(), 1, "{:?}", manifest.checks);
    std::fs::remove_file(&configuration).unwrap();
    let (restored, manifest) = build(&project);
    assert!(manifest.checks.is_empty(), "{:?}", manifest.checks);
    assert_ne!(restored, enabled);
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn new_cargo_configuration_invalidates_cached_kernels() {
    let project = Project::new(HOST);
    project.kernel_crate(KERNEL);
    let (wrapping, manifest) = build(&project);
    assert!(manifest.checks.is_empty(), "{:?}", manifest.checks);
    let configuration = project.0.join(".cargo/config.toml");
    std::fs::create_dir(configuration.parent().unwrap()).unwrap();
    std::fs::write(
        &configuration,
        "[profile.release]\noverflow-checks = true\n",
    )
    .unwrap();
    project.write(&format!("{HOST}\n// host edit\n"));
    let (enabled, manifest) = build(&project);
    assert_ne!(
        enabled, wrapping,
        "new configuration kept wrapping arithmetic"
    );
    assert_eq!(manifest.checks.len(), 1, "{:?}", manifest.checks);
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn workspace_release_profile_invalidates_cached_kernels() {
    let project = Project::new(HOST);
    project.kernel_crate(KERNEL);
    let (wrapping, manifest) = build(&project);
    assert!(manifest.checks.is_empty(), "{:?}", manifest.checks);
    let path = project.0.join("Cargo.toml");
    let manifest = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        format!("{manifest}\n[profile.release]\noverflow-checks = true\n"),
    )
    .unwrap();
    let (enabled, manifest) = build(&project);
    assert_ne!(
        enabled, wrapping,
        "workspace profile kept wrapping arithmetic"
    );
    assert_eq!(manifest.checks.len(), 1, "{:?}", manifest.checks);
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn fetched_packages_keep_their_saved_kernel_profile() {
    let project = Project::new(HOST);
    project.kernel_crate(KERNEL);
    let (wrapping, manifest) = build(&project);
    assert!(manifest.checks.is_empty(), "{:?}", manifest.checks);
    let saved = project
        .0
        .join("metal/src/main.rs/kernels-precise/inputs.json");
    let inputs = std::fs::read(&saved).unwrap();
    std::fs::write(project.0.join(".cargo-ok"), "{\"v\":1}").unwrap();
    project.write(&format!("{HOST}\n// host edit\n"));
    checked(
        project
            .cargo(&["build", "--quiet"])
            .env("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", "true")
            .output()
            .unwrap(),
    );
    assert_eq!(std::fs::read(saved).unwrap(), inputs);
    let manifest =
        Manifest::from_json(&std::fs::read_to_string(wrapping.join("manifest.json")).unwrap())
            .unwrap();
    assert!(manifest.checks.is_empty(), "{:?}", manifest.checks);
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn copied_kernels_leave_out_host_workspace_profiles_and_configuration() {
    let main = KERNEL
        .replace("#![no_std]\n", "")
        .replace(
            "use metal_oxide_device::{DisjointBuffer, kernel, thread_index};",
            "use metal_oxide::device::{DisjointBuffer, thread_index};",
        )
        .replace("#[kernel]", "#[metal_oxide::kernel]")
        + "\nfn main() {}\n";
    let project = Project::new(&main);
    let path = project.0.join("Cargo.toml");
    let manifest = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        path,
        format!("{manifest}\n[profile.release]\noverflow-checks=true\n"),
    )
    .unwrap();
    std::fs::create_dir(project.0.join(".cargo")).unwrap();
    std::fs::write(
        project.0.join(".cargo/config.toml"),
        "[profile.release]\noverflow-checks=true\n",
    )
    .unwrap();
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let directory = project.artifact("add");
    let manifest =
        Manifest::from_json(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert!(manifest.checks.is_empty(), "{:?}", manifest.checks);
}
