#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide::{Device, Error, Module, Pipeline};
use metal_oxide_artifact::{Manifest, sha256};
use std::{path::PathBuf, process::Command};
use support::{Workspace, checked};

fn build(workspace: &Workspace) -> PathBuf {
    let output = checked(workspace.command("build").output().unwrap());
    let log = String::from_utf8(output.stderr).unwrap();
    log.lines()
        .find_map(|line| line.strip_prefix("Built Metal artifact "))
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{log}"))
}

#[test]
#[ignore = "requires a Metal device, compiler nightly, rust-src and Metal toolchain"]
fn embedded_artifacts_load_lazily_in_a_relocated_stable_host() -> metal_oxide::Result<()> {
    let workspace = Workspace::new();
    let source = workspace.0.join("kernels/src/lib.rs");
    let mut kernels = String::from(
        r#"#![no_std]
use metal_oxide_device::{ReadWriteBuffer, block_dim, block_idx, kernel, thread_idx};
#[kernel(block = (8, 1, 1))]
pub unsafe fn in_place(values: ReadWriteBuffer<u64>, increment: u64, n: u32) {
    let i = block_idx().x * block_dim().x + thread_idx().x;
    if i < n { unsafe { values.store_unchecked(i, values.load_unchecked(i).wrapping_add(increment)); } }
}
"#,
    );
    kernels.push_str("mod cooperation {\n");
    kernels.push_str(
        include_str!("../../metal-oxide-compiler/tests/fixtures/dynamic_shared.rs")
            .trim_start_matches("#![no_std]"),
    );
    kernels.push_str("\n}\n");
    for index in 1..483 {
        kernels.push_str(&format!("#[kernel]\npub unsafe fn k{index}() {{}}\n"));
    }
    std::fs::write(&source, &kernels)?;
    let directory = build(&workspace);
    let manifest_text = std::fs::read_to_string(directory.join("manifest.json"))?;
    let mut manifest = Manifest::from_json(&manifest_text)?;
    assert_eq!(manifest.abi.kernels.len(), 484);
    let library = std::fs::read(directory.join("kernels.metallib"))?;
    let bindings = std::fs::read(directory.join("bindings.rs"))?;
    let device = Device::system_default()?;
    let module = Module::from_bytes(&device, &manifest_text, &library)?;
    module.verify_abi(&sha256(manifest.abi.to_json()?.as_bytes()))?;
    Pipeline::new(&device, &module, "in_place")?;
    verify_rejections(&device, &manifest_text, &library);

    std::fs::write(&source, kernels.replace("fn k482()", "fn renamed()"))?;
    let other = build(&workspace);
    let other_library = std::fs::read(other.join("kernels.metallib"))?;
    assert!(matches!(
        Module::from_bytes(&device, &manifest_text, &other_library),
        Err(Error::Artifact(_))
    ));
    // A valid digest with a missing entrypoint must fail when that entry is requested.
    manifest.files.metallib = sha256(&other_library);
    let standalone = workspace.0.join("standalone");
    std::fs::create_dir_all(standalone.join("src"))?;
    let runtime = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../metal-oxide")
        .canonicalize()?;
    std::fs::write(
        standalone.join("Cargo.toml"),
        format!(
            r#"[workspace]
[workspace.dependencies]
metal-oxide = {{ path = {runtime:?} }}
[package]
name = "embedded-consumer"
version = "0.1.0"
edition = "2024"
[dependencies]
metal-oxide = {{ workspace = true }}
"#
        ),
    )?;
    std::fs::write(
        standalone.join("src/main.rs"),
        include_str!("fixtures/embedded_host.rs"),
    )?;
    std::fs::write(standalone.join("manifest.json"), manifest.to_json()?)?;
    std::fs::write(standalone.join("kernels.metallib"), &other_library)?;
    std::fs::write(standalone.join("bindings.rs"), bindings)?;
    checked(
        Command::new("cargo")
            .current_dir(&standalone)
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );
    checked(
        Command::new("cargo")
            .current_dir(&standalone)
            .env("RUSTUP_TOOLCHAIN", "1.89.0")
            .env("RUSTC_BOOTSTRAP", "0")
            .env("CARGO_TARGET_DIR", standalone.join("target"))
            .args(["build", "--release", "--locked", "--offline"])
            .output()
            .unwrap(),
    );
    let moved = workspace.0.join("moved-consumer");
    std::fs::create_dir(&moved)?;
    let binary = moved.join("consumer");
    std::fs::copy(standalone.join("target/release/embedded-consumer"), &binary)?;
    std::fs::remove_dir_all(standalone)?;
    let output = checked(
        Command::new(&binary)
            .current_dir(&moved)
            .env_clear()
            .env("PATH", "")
            .env(
                "METAL_OXIDE_BACKEND",
                std::env::var("METAL_OXIDE_BACKEND").unwrap_or_default(),
            )
            .output()
            .unwrap(),
    );
    let report = String::from_utf8(output.stdout).unwrap();
    assert!(report.contains("embedded consumer passed"), "{report}");
    eprintln!("{report}");
    Ok(())
}

fn verify_rejections(device: &Device, manifest: &str, library: &[u8]) {
    for bytes in [&[][..], &library[..library.len() / 2]] {
        assert!(matches!(
            Module::from_bytes(device, manifest, bytes),
            Err(Error::Artifact(_))
        ));
    }
    assert!(Module::from_bytes(device, "{}", library).is_err());
    let original = serde_json::from_str::<serde_json::Value>(manifest).unwrap();
    for (key, value) in [
        ("version", serde_json::json!(999)),
        ("math_mode", serde_json::json!("unknown")),
        ("required_features", serde_json::json!(["unknown"])),
    ] {
        let mut invalid = original.clone();
        invalid["abi"][key] = value;
        assert!(matches!(
            Module::from_bytes(device, &invalid.to_string(), library),
            Err(Error::Artifact(_))
        ));
    }
}
