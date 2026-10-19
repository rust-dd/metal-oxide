#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide_artifact::Manifest;
use std::path::PathBuf;
use support::{Project, checked, failure};

const VARIABLE: &str = "METAL_OXIDE_TEST_KERNEL_STEP_TWO";
const PRIVATE_VALUE: &str = "kernel-build-environment-value-must-remain-private";
const KERNEL: &str = r#"#![no_std]
use metal_oxide_device::{DisjointBuffer, kernel, thread_index};
#[kernel]
pub fn step(mut out: DisjointBuffer<u32>) {
    const _: bool = option_env!("CARGO_PKG_NAME").is_some();
    const STEP: u32 = if option_env!("METAL_OXIDE_TEST_KERNEL_STEP_TWO").is_some() { 2 } else { 1 };
    if let Some(out) = out.get_mut(&thread_index()) {
        *out = STEP;
    }
}
"#;

fn build(project: &Project, item: &str, enabled: bool) -> (PathBuf, String) {
    let mut command = project.cargo(&["build", "--quiet"]);
    if enabled {
        command.env(VARIABLE, PRIVATE_VALUE);
    } else {
        command.env_remove(VARIABLE);
    }
    checked(command.output().unwrap());
    artifact(project, item)
}

fn artifact(project: &Project, item: &str) -> (PathBuf, String) {
    let saved = project
        .0
        .join("metal/src/main.rs")
        .join(format!("{item}-precise"));
    let manifest =
        Manifest::from_json(&std::fs::read_to_string(saved.join("manifest.json")).unwrap())
            .unwrap();
    let directory = Project::target()
        .join("metal")
        .join(manifest.build.fingerprint);
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    let inputs = std::fs::read_to_string(saved.join("inputs.json")).unwrap();
    assert!(!inputs.contains(PRIVATE_VALUE));
    (directory, msl)
}

fn check_changes(project: &Project, item: &str) {
    let (first, msl) = build(project, item, false);
    assert!(msl.contains("= 1u;"), "{msl}");
    let (second, msl) = build(project, item, true);
    assert_ne!(
        first, second,
        "the host reused a kernel built with another environment"
    );
    assert!(msl.contains("= 2u;"), "{msl}");
    let binary = std::fs::read(Project::target().join("debug/host")).unwrap();
    assert!(
        !binary
            .windows(PRIVATE_VALUE.len())
            .any(|bytes| bytes == PRIVATE_VALUE.as_bytes())
    );
    let (third, msl) = build(project, item, false);
    assert_ne!(second, third);
    assert!(msl.contains("= 1u;"), "{msl}");
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn referenced_environment_changes_rebuild_kernel_crates_without_a_source_edit() {
    let project =
        Project::new("#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\nfn main() {}\n");
    project.kernel_crate(KERNEL);
    check_changes(&project, "kernels");
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn referenced_environment_changes_rebuild_copied_kernels_without_a_source_edit() {
    let project = Project::new(&copied_source());
    check_changes(&project, "step");
}

fn copied_source() -> String {
    KERNEL
        .replace("#![no_std]\n", "")
        .replace(
            "use metal_oxide_device::{DisjointBuffer, kernel, thread_index};",
            "use metal_oxide::device::{DisjointBuffer, thread_index};",
        )
        .replace("#[kernel]", "#[metal_oxide::kernel]")
        + "\nfn main() {}\n"
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn forced_configuration_tracks_nested_values_instead_of_host_values() {
    let project = Project::new(&copied_source());
    build(&project, "step", false);
    let state = project.state("step", metal_oxide_artifact::MathMode::Precise, false);
    let compiler = state["compiler"].as_str().unwrap();
    let target = project.0.join("nested-target");
    for (root, value) in [(&project.0, "host"), (&target, "different-device-value")] {
        std::fs::create_dir_all(root.join(".cargo")).unwrap();
        std::fs::write(
            root.join(".cargo/config.toml"),
            format!("[env]\n{VARIABLE}={{value={value:?},force=true}}\n"),
        )
        .unwrap();
    }
    let build = |locked: bool| {
        let mut command = project.cargo(&["build", "--quiet"]);
        command.env("CARGO_TARGET_DIR", &target);
        command.env("METAL_OXIDE_COMPILER", compiler);
        if locked {
            command.env("METAL_OXIDE_ARTIFACTS", "locked");
        }
        checked(command.output().unwrap());
    };
    build(false);
    build(true);
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn nested_configuration_defaults_track_ambient_overrides_without_becoming_stale() {
    let project =
        Project::new("#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\nfn main() {}\n");
    let kernel = KERNEL.replace(
        "if option_env!(\"METAL_OXIDE_TEST_KERNEL_STEP_TWO\").is_some() { 2 } else { 1 }",
        "env!(\"METAL_OXIDE_TEST_KERNEL_STEP_TWO\").as_bytes()[0] as u32 - 48",
    );
    project.kernel_crate(&kernel);
    let manifest = project.0.join("Cargo.toml");
    let host = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(manifest, host.replace("members = [\"kernels\"]\n", "")).unwrap();
    let manifest = project.0.join("kernels/Cargo.toml");
    let kernel = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(manifest, kernel + "\n[workspace]\n").unwrap();
    checked(
        std::process::Command::new("cargo")
            .current_dir(project.0.join("kernels"))
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );
    std::fs::create_dir(project.0.join("kernels/.cargo")).unwrap();
    std::fs::write(
        project.0.join("kernels/.cargo/config.toml"),
        format!("[env]\n{VARIABLE}='2'\n"),
    )
    .unwrap();
    for ambient in [None, Some("1"), None] {
        for locked in [false, true] {
            let mut command = project.cargo(&["build", "--quiet"]);
            if let Some(value) = ambient {
                command.env(VARIABLE, value);
            } else {
                command.env_remove(VARIABLE);
            }
            if locked {
                command.env("METAL_OXIDE_ARTIFACTS", "locked");
            }
            checked(command.output().unwrap());
            let (_, msl) = artifact(&project, "kernels");
            let expected = if ambient.is_some() { "= 1u;" } else { "= 2u;" };
            assert!(msl.contains(expected), "{msl}");
        }
    }
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn local_locked_artifacts_check_environment_while_fetched_artifacts_keep_their_values() {
    let project =
        Project::new("#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\nfn main() {}\n");
    project.kernel_crate(KERNEL);
    build(&project, "kernels", false);
    let saved = project
        .0
        .join("metal/src/main.rs/kernels-precise/inputs.json");
    let inputs = std::fs::read(&saved).unwrap();
    let locked = |enabled: bool| {
        let mut command = project.cargo(&["build", "--quiet"]);
        command.env("METAL_OXIDE_ARTIFACTS", "locked");
        if enabled {
            command.env(VARIABLE, PRIVATE_VALUE);
        } else {
            command.env_remove(VARIABLE);
        }
        command.output().unwrap()
    };
    checked(locked(false));
    let errors = failure(locked(true));
    assert!(errors.contains("is stale"), "{errors}");
    std::fs::write(project.0.join(".cargo-ok"), "{\"v\":1}").unwrap();
    checked(locked(true));
    assert_eq!(std::fs::read(saved).unwrap(), inputs);
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn host_build_scripts_can_supply_referenced_kernel_environment() {
    let project =
        Project::new("#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\nfn main() {}\n");
    project.kernel_crate(KERNEL);
    const TRIGGER: &str = "METAL_OXIDE_TEST_HOST_STEP_TWO";
    std::fs::write(
        project.0.join("build.rs"),
        format!(
            r#"fn main() {{
    println!("cargo:rerun-if-env-changed={TRIGGER}");
    if std::env::var_os("{TRIGGER}").is_some() {{
        println!("cargo:rustc-env={VARIABLE}={PRIVATE_VALUE}");
    }}
}}
"#
        ),
    )
    .unwrap();
    for enabled in [false, true, false] {
        let mut command = project.cargo(&["build", "--quiet"]);
        command.env_remove(VARIABLE);
        if enabled {
            command.env(TRIGGER, "1");
        } else {
            command.env_remove(TRIGGER);
        }
        checked(command.output().unwrap());
        let (_, msl) = artifact(&project, "kernels");
        let expected = if enabled { "= 2u;" } else { "= 1u;" };
        assert!(msl.contains(expected), "{msl}");
    }
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn isolated_tool_variables_are_validated_as_kernel_context_not_host_environment() {
    let main = copied_source().replace(VARIABLE, "RUSTFLAGS");
    let project = Project::new(&main);
    let build = |locked: bool| {
        let mut command = project.cargo(&["build", "--quiet"]);
        command.env("RUSTFLAGS", "--cfg host_only");
        if locked {
            command.env("METAL_OXIDE_ARTIFACTS", "locked");
        }
        checked(command.output().unwrap());
    };
    build(false);
    build(true);
}
