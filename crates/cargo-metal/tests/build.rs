mod support;

use support::{Workspace, checked};

#[test]
#[ignore = "requires the compiler nightly and rust-src"]
fn device_build_ignores_host_only_dependencies() {
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
}
