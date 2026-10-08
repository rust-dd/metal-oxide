mod support;

use metal_oxide_artifact::{Abi, Manifest, sha256};
use std::{
    fs::{File, OpenOptions},
    process::Stdio,
    time::{Duration, Instant},
};
use support::{Workspace, checked};

#[test]
#[ignore = "requires the compiler nightly and rust-src"]
fn device_build_validates_only_selected_dependencies() {
    let workspace = Workspace::new();
    let output = checked(
        workspace
            .command("inspect")
            .args(["--emit", "msl"])
            .output()
            .unwrap(),
    );
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("kernel void empty")
    );

    let manifest = workspace.0.join("kernels/Cargo.toml");
    let source = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(
        &manifest,
        source.replace("not(target_env = \"metal\")", "target_env = \"metal\""),
    )
    .unwrap();
    let output = workspace
        .command("inspect")
        .args(["--emit", "msl"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("kernel build scripts are unsupported: host-only"),
        "{error}"
    );
    assert!(
        !error.contains("host-only build script must not execute"),
        "{error}"
    );

    let source = source.split("[target.").next().unwrap();
    let source = format!(
        "{source}host-only={{path=\"../host-only\",optional=true}}\n[features]\nextra=[\"dep:host-only\"]\n"
    );
    std::fs::write(&manifest, &source).unwrap();
    let host = workspace.0.join("host/Cargo.toml");
    let host_source = std::fs::read_to_string(&host).unwrap();
    std::fs::write(host, format!("{host_source}[dependencies]\ntest-kernels={{path=\"../kernels\",features=[\"extra\"]}}\n")).unwrap();
    checked(
        std::process::Command::new("cargo")
            .current_dir(&workspace.0)
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );
    checked(
        workspace
            .command("inspect")
            .args(["--emit", "msl"])
            .output()
            .unwrap(),
    );

    std::fs::write(manifest, format!("{source}default=[\"extra\"]\n")).unwrap();
    let output = workspace
        .command("inspect")
        .args(["--emit", "msl"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("kernel build scripts are unsupported: host-only"),
        "{error}"
    );
    assert!(
        !error.contains("host-only build script must not execute"),
        "{error}"
    );
}

#[test]
#[ignore = "requires the compiler nightly and rust-src"]
fn inconsistent_cargo_package_selection_stops_before_build_scripts() {
    let workspace = Workspace::new();
    checked(
        workspace
            .command("inspect")
            .args(["--emit", "msl"])
            .env_remove("METAL_OXIDE_COMPILER")
            .output()
            .unwrap(),
    );
    let compiler = workspace
        .0
        .join("target/compiler/debug/metal-oxide-compiler");
    let manifest = workspace.0.join("Cargo.toml");
    let source = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(manifest, format!("{source}exclude=[\"safe\",\"active\"]\n")).unwrap();
    for name in ["safe", "active"] {
        let directory = workspace.0.join(name);
        std::fs::create_dir_all(directory.join("src")).unwrap();
        std::fs::write(
            directory.join("Cargo.toml"),
            "[package]\nname=\"probe-helper\"\nversion=\"0.1.0\"\nedition=\"2024\"\n",
        )
        .unwrap();
        std::fs::write(directory.join("src/lib.rs"), "#![no_std]\n").unwrap();
    }
    std::fs::write(
        workspace.0.join("active/build.rs"),
        "fn main() { panic!(\"probe-helper build script must not execute\") }\n",
    )
    .unwrap();
    for (directory, path) in [
        (workspace.0.clone(), "active"),
        (workspace.0.join("host"), "../safe"),
    ] {
        let config = directory.join(".cargo");
        std::fs::create_dir_all(&config).unwrap();
        std::fs::write(
            config.join("config.toml"),
            format!("[patch.crates-io]\nprobe-helper={{path=\"{path}\"}}\n"),
        )
        .unwrap();
    }
    let manifest = workspace.0.join("kernels/Cargo.toml");
    let source = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(
        manifest,
        source.replace("[dependencies]", "[dependencies]\nprobe-helper=\"0.1.0\""),
    )
    .unwrap();
    checked(
        std::process::Command::new("cargo")
            .current_dir(&workspace.0)
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );
    let output = workspace
        .command("inspect")
        .args(["--emit", "msl"])
        .env("METAL_OXIDE_COMPILER", compiler)
        .current_dir(workspace.0.join("host"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("missing Cargo package for") && error.contains("/active#probe-helper@0.1.0"),
        "{error}"
    );
    assert!(
        !error.contains("probe-helper build script must not execute"),
        "{error}"
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn concurrent_builds_wait_before_writing_and_publish_valid_artifacts() {
    let workspace = Workspace::new();
    let root = workspace.0.join("target/metal");
    std::fs::create_dir_all(&root).unwrap();
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(".lock"))
        .unwrap();
    lock.lock().unwrap();
    let mut children = ["true", "false"].map(|overflow| {
        let log = workspace.0.join(format!("{overflow}.log"));
        let child = workspace
            .command("build")
            .env("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", overflow)
            .stdout(Stdio::null())
            .stderr(File::create(&log).unwrap())
            .spawn()
            .unwrap();
        (child, log)
    });
    let deadline = Instant::now() + Duration::from_secs(20);
    let blocked = loop {
        if children.iter().all(|(_, log)| {
            std::fs::read_to_string(log)
                .unwrap()
                .contains("Blocking waiting for Metal build lock")
        }) {
            break true;
        }
        if Instant::now() >= deadline
            || children
                .iter_mut()
                .all(|(child, _)| child.try_wait().unwrap().is_some())
        {
            break false;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let untouched = !root.join("metal64-unknown-none.json").exists();
    drop(lock);
    let results = children
        .into_iter()
        .map(|(mut child, log)| (child.wait().unwrap(), std::fs::read_to_string(log).unwrap()))
        .collect::<Vec<_>>();
    let mut directories = Vec::new();
    for (status, log) in results {
        assert!(status.success(), "{log}");
        let directory = log
            .lines()
            .find_map(|line| line.strip_prefix("Built Metal artifact "))
            .unwrap();
        directories.push(std::path::PathBuf::from(directory));
    }
    assert!(blocked, "builds did not wait for the shared Metal output");
    assert!(
        untouched,
        "compiler output changed while another build held the lock"
    );
    assert_ne!(directories[0], directories[1]);
    for directory in directories {
        let manifest =
            Manifest::from_json(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            manifest.build.fingerprint,
            directory.file_name().unwrap().to_str().unwrap()
        );
        assert_eq!(
            manifest.abi,
            Abi::from_json(&std::fs::read_to_string(directory.join("abi.json")).unwrap()).unwrap()
        );
        for (file, expected) in manifest.files.entries() {
            assert_eq!(
                sha256(&std::fs::read(directory.join(file.name())).unwrap()),
                expected
            );
        }
    }
}
