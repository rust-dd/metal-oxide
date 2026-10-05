use std::{path::PathBuf, process::Command};

const COMPILER: &str = env!("CARGO_BIN_EXE_metal-oxide-compiler");

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn compile(source: &str, emit: &str) -> std::process::Output {
    let directory = root().join("target/frontend-tests");
    std::fs::create_dir_all(&directory).unwrap();
    Command::new(COMPILER)
        .arg(
            root()
                .join("crates/metal-oxide-compiler/tests/fixtures")
                .join(source),
        )
        .args(["--crate-type", "rlib", "--edition", "2024", "--emit", emit])
        .arg("--target")
        .arg(root().join("targets/metal64-unknown-none.json"))
        .args(["-Z", "unstable-options"])
        .arg("-o")
        .arg(directory.join(format!("{source}.{emit}")))
        .output()
        .unwrap()
}

#[test]
fn device_target_has_explicit_scalar_and_pointer_layout() {
    let output = compile("empty.rs", "metadata");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = String::from_utf8(output.stdout).unwrap();
    assert!(
        report.contains("pointer=64 usize=64 endian=little"),
        "{report}"
    );
    assert!(report.contains("f32: size=4 align=4"), "{report}");
    assert!(report.contains("u32: size=4 align=4"), "{report}");
    assert!(report.contains("i32: size=4 align=4"), "{report}");
}

#[test]
fn native_output_is_rejected() {
    let output = compile("empty.rs", "obj");
    assert!(!output.status.success());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.contains("native code generation is not supported"),
        "{error}"
    );
}
