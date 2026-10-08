use super::Inputs;
use crate::{
    cli::{Action, Options},
    metadata, process,
};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-inputs-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        for name in ["host", "kernels", "helper", "unused"] {
            std::fs::create_dir_all(path.join(name).join("src")).unwrap();
        }
        let path = path.canonicalize().unwrap();
        std::fs::write(path.join("Cargo.toml"),
            "[workspace]\nmembers=[\"host\",\"kernels\",\"helper\",\"unused\"]\nresolver=\"3\"\n[workspace.dependencies]\ninput-helper={path=\"helper\"}\nunused-helper={path=\"unused\"}\n").unwrap();
        for (directory, name, settings) in [
            (
                "host",
                "input-host",
                "[package.metadata.metal]\nkernels=\"../kernels/Cargo.toml\"\n",
            ),
            (
                "kernels",
                "input-kernels",
                "[dependencies]\ninput-helper={workspace=true}\n[dev-dependencies]\nunused-helper={workspace=true}\n",
            ),
            ("helper", "input-helper", ""),
            ("unused", "unused-helper", ""),
        ] {
            std::fs::write(
                path.join(directory).join("Cargo.toml"),
                format!(
                    "[package]\nname=\"{name}\"\nversion=\"0.1.0\"\nedition=\"2024\"\n{settings}"
                ),
            )
            .unwrap();
        }
        std::fs::write(path.join("host/src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            path.join("kernels/src/lib.rs"),
            "#![no_std]\npub const VALUE: u32 = input_helper::VALUE + 1;\n",
        )
        .unwrap();
        for name in ["helper", "unused"] {
            std::fs::write(
                path.join(name).join("src/lib.rs"),
                "#![no_std]\npub const VALUE: u32 = 10;\n",
            )
            .unwrap();
        }
        process::capture(
            Command::new("cargo")
                .current_dir(&path)
                .args(["generate-lockfile", "--offline"]),
        )
        .unwrap();
        Self(path)
    }

    fn inputs(&self) -> Inputs {
        let output = process::capture(
            Command::new("cargo")
                .current_dir(&self.0)
                .env("CARGO_TARGET_DIR", self.0.join("target"))
                .args([
                    "check",
                    "--lib",
                    "--locked",
                    "--offline",
                    "-p",
                    "input-kernels",
                    "--message-format=json-render-diagnostics",
                ]),
        )
        .unwrap();
        let project = metadata::load(&Options {
            action: Action::Inspect,
            manifest: self.0.join("Cargo.toml"),
            package: Some("input-host".into()),
            release: false,
            arguments: vec![],
        })
        .unwrap();
        Inputs::from_cargo(&output, &project).unwrap()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn unrelated_readmes_do_not_change_kernel_inputs() {
    let workspace = Workspace::new();
    let before = workspace.inputs();
    for name in ["kernels", "helper"] {
        std::fs::write(workspace.0.join(name).join("README.md"), "Changed prose\n").unwrap();
    }
    assert_eq!(before, workspace.inputs());
}

#[cfg(unix)]
#[test]
fn unrelated_symlinks_do_not_reject_kernel_inputs() {
    let workspace = Workspace::new();
    let before = workspace.inputs();
    std::os::unix::fs::symlink(
        "missing-not-a-build-input",
        workspace.0.join("helper/reference"),
    )
    .unwrap();
    assert_eq!(before, workspace.inputs());
}

#[test]
fn kernel_and_dependency_changes_invalidate_inputs() {
    let workspace = Workspace::new();
    let before = workspace.inputs();
    std::fs::write(
        workspace.0.join("kernels/src/lib.rs"),
        "#![no_std]\npub const VALUE: u32 = input_helper::VALUE + 2;\n",
    )
    .unwrap();
    let kernel_changed = workspace.inputs();
    assert_ne!(before, kernel_changed);
    std::fs::write(
        workspace.0.join("helper/src/lib.rs"),
        "#![no_std]\npub const VALUE: u32 = 99;\n",
    )
    .unwrap();
    assert_ne!(kernel_changed, workspace.inputs());
}

#[test]
fn unused_dev_dependency_changes_do_not_change_inputs() {
    let workspace = Workspace::new();
    let before = workspace.inputs();
    std::fs::write(
        workspace.0.join("unused/src/lib.rs"),
        "#![no_std]\npub const VALUE: u32 = 99;\n",
    )
    .unwrap();
    assert_eq!(before, workspace.inputs());
}

#[test]
fn fixture_build_outputs_are_isolated_from_inherited_cargo_targets() {
    let workspace = Workspace::new();
    let inputs = workspace.inputs();
    let outputs = inputs
        .files
        .keys()
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "rmeta")
        })
        .collect::<Vec<_>>();
    assert!(!outputs.is_empty());
    assert!(
        outputs
            .iter()
            .all(|path| path.starts_with(workspace.0.join("target")))
    );
}

#[test]
fn feature_lockfile_and_configuration_changes_invalidate_inputs() {
    let workspace = Workspace::new();
    let before = workspace.inputs();
    let manifest = workspace.0.join("kernels/Cargo.toml");
    let source = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(
        manifest,
        format!("{source}[features]\nextra=[]\ndefault=[\"extra\"]\n"),
    )
    .unwrap();
    let features = workspace.inputs();
    assert_ne!(before, features);
    assert!(features.features.values().any(|set| set.contains("extra")));
    let lock = workspace.0.join("Cargo.lock");
    let source = std::fs::read_to_string(&lock).unwrap();
    std::fs::write(lock, format!("# input identity regression\n{source}")).unwrap();
    let locked = workspace.inputs();
    assert_ne!(features, locked);
    std::fs::create_dir(workspace.0.join(".cargo")).unwrap();
    std::fs::write(
        workspace.0.join(".cargo/config.toml"),
        "[build]\nrustflags=[\"--cfg\",\"cache_probe\"]\n",
    )
    .unwrap();
    assert_ne!(locked, workspace.inputs());
}
