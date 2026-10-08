#[allow(dead_code)]
mod support;

use std::{os::unix::fs::PermissionsExt, path::Path, process::Command};
use support::{Workspace, checked};

fn install_cli(workspace: &Workspace) -> std::path::PathBuf {
    let directory = workspace.0.join("installed/bin");
    std::fs::create_dir_all(&directory).unwrap();
    let cli = directory.join("cargo-metal");
    std::fs::copy(env!("CARGO_BIN_EXE_cargo-metal"), &cli).unwrap();
    cli
}

fn compiler(path: &Path, changes: serde_json::Value) {
    let rustc = checked(
        Command::new("rustup")
            .args(["run", "nightly-2026-10-04", "rustc", "-vV"])
            .output()
            .unwrap(),
    );
    let info = serde_json::json!({"protocol": 1, "abi": 3, "version": env!("CARGO_PKG_VERSION"), "rustc": String::from_utf8(rustc.stdout).unwrap().trim(), "binary": ""});
    std::fs::write(path, format!("#!/usr/bin/env python3\nimport hashlib,json,pathlib\nv=json.loads({:?})\nv.update(json.loads({:?}))\nv['binary']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()\nprint(json.dumps(v))\n", info.to_string(), changes.to_string())).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
#[ignore = "requires pinned nightly and Python"]
fn installed_cli_requires_a_compatible_adjacent_compiler() {
    let workspace = Workspace::new();
    let cli = install_cli(&workspace);
    let run = || {
        Command::new(&cli)
            .current_dir(&workspace.0)
            .env_remove("METAL_OXIDE_COMPILER")
            .env("CARGO_TARGET_DIR", workspace.0.join("target"))
            .args(["inspect", "-p", "test-host", "--emit", "msl"])
            .output()
            .unwrap()
    };
    let missing = run();
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("no installed metal-oxide-compiler"));
    let adjacent = cli.with_file_name("metal-oxide-compiler");
    for (field, value) in [
        ("version", serde_json::json!("wrong")),
        ("protocol", 99.into()),
        ("abi", 99.into()),
        ("rustc", "wrong nightly".into()),
    ] {
        compiler(&adjacent, serde_json::json!({field: value}));
        let output = run();
        assert!(!output.status.success());
        let message = String::from_utf8_lossy(&output.stderr);
        assert!(
            message.contains(&format!("compiler {field} mismatch")),
            "{message}"
        );
    }
    let override_path = workspace.0.join("override-compiler");
    compiler(&override_path, serde_json::json!({"version": "override"}));
    let output = Command::new(&cli)
        .current_dir(&workspace.0)
        .env("METAL_OXIDE_COMPILER", &override_path)
        .args(["inspect", "-p", "test-host", "--emit", "msl"])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&output.stderr).contains("override"));
}

#[test]
#[ignore = "requires pinned nightly and Python"]
fn doctor_reports_missing_components_with_the_install_command() {
    let workspace = Workspace::new();
    let cli = install_cli(&workspace);
    compiler(
        &cli.with_file_name("metal-oxide-compiler"),
        serde_json::json!({}),
    );
    let directory = workspace.0.join("probes");
    std::fs::create_dir(&directory).unwrap();
    let path = std::env::var_os("PATH").unwrap();
    let rustup = std::env::split_paths(&path)
        .map(|p| p.join("rustup"))
        .find(|p| p.is_file())
        .unwrap();
    let shim = directory.join("rustup");
    std::fs::write(&shim, format!("#!/usr/bin/env python3\nimport os,sys\nif sys.argv[1:3]==['component','list']:\n print('rustc-dev\\nllvm-tools')\nelse:\n os.execv({:?}, [{:?}] + sys.argv[1:])\n", rustup.to_str().unwrap(), rustup.to_str().unwrap())).unwrap();
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    let output = Command::new(cli)
        .arg("doctor")
        .env_remove("METAL_OXIDE_COMPILER")
        .env(
            "PATH",
            std::env::join_paths(std::iter::once(directory).chain(std::env::split_paths(&path)))
                .unwrap(),
        )
        .output()
        .unwrap();
    assert!(!output.status.success());
    let message = String::from_utf8_lossy(&output.stdout);
    assert!(message.contains("is missing rust-src"), "{message}");
    assert!(message.contains("rustup toolchain install nightly-2026-10-04 --component rustc-dev --component rust-src --component llvm-tools"), "{message}");
    assert!(message.contains("[ok] metal-oxide compiler:"), "{message}");
}

#[test]
#[ignore = "requires pinned nightly, rustc-dev and rust-src"]
fn copied_cli_and_compiler_build_an_external_kernel() {
    let workspace = Workspace::new();
    checked(
        workspace
            .command("inspect")
            .args(["--emit", "msl"])
            .env_remove("METAL_OXIDE_COMPILER")
            .output()
            .unwrap(),
    );
    let cli = install_cli(&workspace);
    std::fs::copy(
        workspace
            .0
            .join("target/compiler/debug/metal-oxide-compiler"),
        cli.with_file_name("metal-oxide-compiler"),
    )
    .unwrap();
    let output = checked(
        Command::new(cli)
            .current_dir(&workspace.0)
            .env_remove("METAL_OXIDE_COMPILER")
            .env("CARGO_TARGET_DIR", workspace.0.join("installed-build"))
            .args([
                "inspect",
                "--manifest-path",
                "host/Cargo.toml",
                "-p",
                "test-host",
                "--emit",
                "msl",
            ])
            .output()
            .unwrap(),
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("kernel void empty"));
    assert!(!workspace.0.join("installed-build/compiler").exists());
}
