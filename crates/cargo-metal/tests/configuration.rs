mod support;

use support::{Workspace, checked};

#[test]
#[ignore = "requires the compiler nightly and rust-src"]
fn cargo_resolves_configured_flags_and_environment_precedence() {
    let workspace = Workspace::new();
    let config = workspace.0.join(".cargo/config.toml");
    std::fs::create_dir_all(config.parent().unwrap()).unwrap();
    std::fs::write(
        workspace.0.join("kernels/src/lib.rs"),
        r#"
#![no_std]
use metal_oxide_device::kernel;
#[cfg(not(any(audit = "build", audit = "target", audit = "environment", audit = "encoded")))]
compile_error!("kernel compilation lost configured flags");
#[cfg(audit = "build")]
#[kernel] pub unsafe fn from_build() {}
#[cfg(audit = "target")]
#[kernel] pub unsafe fn from_target() {}
#[cfg(audit = "environment")]
#[kernel] pub unsafe fn from_environment() {}
#[cfg(audit = "encoded")]
#[kernel] pub unsafe fn from_encoded() {}
"#,
    )
    .unwrap();
    let build = "[build]\nrustflags=['--cfg', 'audit=\"build\"']\n";
    let target =
        "[target.'cfg(target_env = \"metal\")']\nrustflags=['--cfg', 'audit=\"target\"']\n";
    for (settings, flags, encoded, expected) in [
        (build.to_owned(), None, None, "build"),
        (format!("{build}{target}"), None, None, "target"),
        (
            format!("{build}{target}"),
            Some("--cfg audit=\"environment\""),
            None,
            "environment",
        ),
        (
            build.to_owned(),
            Some("--cfg audit=\"environment\""),
            Some("--cfg\u{1f}audit=\"encoded\""),
            "encoded",
        ),
        (build.to_owned(), None, None, "build"),
    ] {
        std::fs::write(&config, settings).unwrap();
        let mut command = workspace.command("inspect");
        command
            .args(["--emit", "msl"])
            .env_remove("RUSTFLAGS")
            .env_remove("CARGO_ENCODED_RUSTFLAGS");
        if let Some(flags) = flags {
            command.env("RUSTFLAGS", flags);
        }
        if let Some(encoded) = encoded {
            command.env("CARGO_ENCODED_RUSTFLAGS", encoded);
        }
        let output = checked(command.output().unwrap());
        let source = String::from_utf8(output.stdout).unwrap();
        assert!(
            source.contains(&format!("kernel void from_{expected}")),
            "{source}"
        );
        let rust = std::fs::read_dir(workspace.0.join("target/metal/rust"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let output = std::fs::read_dir(rust)
            .unwrap()
            .map(|entry| entry.unwrap())
            .find(|entry| entry.file_name().to_string_lossy().starts_with("output-"))
            .unwrap()
            .path();
        let arguments: Vec<String> =
            serde_json::from_slice(&std::fs::read(output.join("rustc-args.json")).unwrap())
                .unwrap();
        assert!(
            arguments.contains(&format!("audit=\"{expected}\"")),
            "{arguments:?}"
        );
    }
}

#[test]
#[ignore = "requires the compiler nightly and rust-src"]
fn development_compiler_is_built_for_the_host_despite_a_cargo_target() {
    let workspace = Workspace::new();
    let stale = workspace
        .0
        .join("target/compiler/debug/metal-oxide-compiler");
    std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
    std::fs::write(&stale, b"stale compiler at the old default path").unwrap();
    let output = checked(
        workspace
            .command("inspect")
            .args(["--emit", "msl"])
            .env_remove("METAL_OXIDE_COMPILER")
            .env("CARGO_BUILD_TARGET", "wasm32-unknown-unknown")
            .output()
            .unwrap(),
    );
    assert!(
        String::from_utf8(output.stdout)
            .unwrap()
            .contains("kernel void empty")
    );
    assert!(workspace.compiler().is_file());
    assert_eq!(
        std::fs::read(stale).unwrap(),
        b"stale compiler at the old default path"
    );
}
