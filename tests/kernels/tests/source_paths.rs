#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide_artifact::{CheckKind, Manifest};
use std::{path::Path, process::Command};
use support::{Project, checked};

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn dependency_checks_have_portable_paths_and_capture_keeps_local_sources() {
    let project =
        Project::new("#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\nfn main() {}\n");
    project.kernel_crate(
        "#![no_std]\nuse metal_oxide_device::*;\n#[kernel]\npub fn divide(mut out: DisjointBuffer<u32>, value: u32, divisor: u32) {\n    if let Some(out) = out.get_mut(&thread_index()) {\n        *out = checked_helper::divide(value, divisor);\n    }\n}\n",
    );
    let dependency = Project::new("fn main() {}\n");
    let helper = &dependency.0;
    std::fs::create_dir_all(helper.join("src")).unwrap();
    std::fs::write(
        helper.join("Cargo.toml"),
        "[package]\nname = \"checked-helper\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[workspace]\n",
    )
    .unwrap();
    std::fs::write(
        helper.join("src/lib.rs"),
        "#![no_std]\npub fn divide(value: u32, divisor: u32) -> u32 {\n    value / divisor\n}\n",
    )
    .unwrap();
    let manifest = project.0.join("kernels/Cargo.toml");
    let text = std::fs::read_to_string(&manifest).unwrap()
        + &format!(
            "checked-helper = {{ path = {:?} }}\n",
            helper.display().to_string()
        );
    std::fs::write(manifest, text).unwrap();
    checked(
        Command::new("cargo")
            .current_dir(&project.0)
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );
    let build = |capture| {
        let mut command = project.cargo(&["build", "--quiet"]);
        if capture {
            command.env("METAL_OXIDE_CAPTURE", project.0.join("capture.gputrace"));
        }
        let built = project.built(&mut command);
        let text = std::fs::read_to_string(built["kernels"].join("manifest.json")).unwrap();
        Manifest::from_json(&text).unwrap()
    };
    let release = build(false);
    let check = release
        .checks
        .iter()
        .find(|check| check.kind == CheckKind::DivisionByZero)
        .unwrap();
    assert_eq!(check.location.line, 3);
    assert!(!Path::new(&check.location.file).is_absolute(), "{check:?}");
    assert_eq!(
        check.location.file,
        "dependencies/checked-helper-0.1.0/src/lib.rs"
    );
    let saved = project
        .0
        .join("metal/src/main.rs/kernels-precise/manifest.json");
    let saved = std::fs::read(saved).unwrap();
    let capture = build(true);
    let check = capture
        .checks
        .iter()
        .find(|check| check.kind == CheckKind::DivisionByZero)
        .unwrap();
    assert_eq!(Path::new(&check.location.file), helper.join("src/lib.rs"));
    assert_eq!(
        std::fs::read(
            project
                .0
                .join("metal/src/main.rs/kernels-precise/manifest.json")
        )
        .unwrap(),
        saved,
    );
}
