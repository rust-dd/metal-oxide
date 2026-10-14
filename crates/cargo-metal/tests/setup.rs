mod support;

use std::{os::unix::fs::PermissionsExt, path::Path, process::Command};
use support::checked;

/// A compiler stand-in that reports `changes` over a matching identity.
fn compiler(path: &Path, changes: serde_json::Value) {
    let rustc = checked(
        Command::new("rustup")
            .args([
                "run",
                metal_oxide_artifact::COMPILER_NIGHTLY,
                "rustc",
                "-vV",
            ])
            .output()
            .unwrap(),
    );
    let info = serde_json::json!({"protocol": 1, "abi": metal_oxide_artifact::ABI_VERSION, "version": env!("CARGO_PKG_VERSION"), "rustc": String::from_utf8(rustc.stdout).unwrap().trim(), "binary": ""});
    std::fs::write(path, format!("#!/usr/bin/env python3\nimport hashlib,json,pathlib\nv=json.loads({:?})\nv.update(json.loads({:?}))\nv['binary']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()\nprint(json.dumps(v))\n", info.to_string(), changes.to_string())).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
#[ignore = "requires the pinned nightly and Python"]
fn setup_check_reports_missing_components_and_mismatched_compilers() {
    let directory = support::root().join(format!("target/cli-tests/setup-{}", std::process::id()));
    let bin = directory.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let cli = bin.join("cargo-metal");
    std::fs::copy(env!("CARGO_BIN_EXE_cargo-metal"), &cli).unwrap();
    let path = std::env::var_os("PATH").unwrap();
    let rustup = std::env::split_paths(&path)
        .map(|p| p.join("rustup"))
        .find(|p| p.is_file())
        .unwrap();
    let probes = directory.join("probes");
    std::fs::create_dir(&probes).unwrap();
    let shim = probes.join("rustup");
    std::fs::write(&shim, format!("#!/usr/bin/env python3\nimport os,sys\nif sys.argv[1:3]==['component','list']:\n print('rustc-dev\\nllvm-tools')\nelse:\n os.execv({:?}, [{:?}] + sys.argv[1:])\n", rustup.to_str().unwrap(), rustup.to_str().unwrap())).unwrap();
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    let check = || {
        Command::new(&cli)
            .args(["metal", "setup", "--check"])
            .env_remove("METAL_OXIDE_COMPILER")
            .env(
                "PATH",
                std::env::join_paths(
                    std::iter::once(probes.clone()).chain(std::env::split_paths(&path)),
                )
                .unwrap(),
            )
            .output()
            .unwrap()
    };
    compiler(&bin.join("metal-oxide-compiler"), serde_json::json!({}));
    let output = check();
    assert!(!output.status.success());
    let report = String::from_utf8_lossy(&output.stdout);
    assert!(report.contains("is missing rust-src"), "{report}");
    assert!(report.contains("[ok] metal-oxide compiler:"), "{report}");
    assert!(report.contains("cargo metal setup"), "{report}");

    for (field, value) in [
        ("version", serde_json::json!("wrong")),
        ("abi", 99.into()),
        ("rustc", "wrong nightly".into()),
    ] {
        compiler(
            &bin.join("metal-oxide-compiler"),
            serde_json::json!({field: value}),
        );
        let report = String::from_utf8_lossy(&check().stdout).into_owned();
        assert!(
            report.contains(&format!("compiler {field} mismatch")),
            "{report}"
        );
    }
    std::fs::remove_dir_all(&directory).unwrap();
}
