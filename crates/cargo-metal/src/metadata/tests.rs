use super::*;
use crate::cli::Action;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Workspace(PathBuf);

impl Workspace {
    fn new(kind: &str, proc_macro: bool) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-dependencies-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        for name in ["host", "kernels", "test-support"] {
            std::fs::create_dir_all(path.join(name).join("src")).unwrap();
        }
        let path = path.canonicalize().unwrap();
        std::fs::write(
            path.join("Cargo.toml"),
            "[workspace]\nmembers=[\"host\",\"kernels\",\"test-support\"]\nresolver=\"3\"\n",
        )
        .unwrap();
        std::fs::write(path.join("host/Cargo.toml"),
            "[package]\nname=\"audit-host\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[package.metadata.metal]\nkernels=\"../kernels/Cargo.toml\"\n").unwrap();
        std::fs::write(path.join("host/src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(path.join("kernels/Cargo.toml"), format!(
            "[package]\nname=\"audit-kernels\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[{kind}]\naudit-test-support={{path=\"../test-support\"}}\n")).unwrap();
        std::fs::write(path.join("kernels/src/lib.rs"), "#![no_std]\n").unwrap();
        let target = if proc_macro {
            "[lib]\nproc-macro=true\n"
        } else {
            ""
        };
        std::fs::write(path.join("test-support/Cargo.toml"), format!(
            "[package]\nname=\"audit-test-support\"\nversion=\"0.1.0\"\nedition=\"2024\"\n{target}")).unwrap();
        std::fs::write(path.join("test-support/src/lib.rs"), "").unwrap();
        if !proc_macro {
            std::fs::write(
                path.join("test-support/build.rs"),
                "fn main() { panic!(\"must not execute during metadata selection\") }\n",
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

    fn options(&self) -> Options {
        Options {
            action: Action::Inspect,
            manifest: self.0.join("Cargo.toml"),
            package: Some("audit-host".into()),
            release: false,
            arguments: vec![],
        }
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn unused_dev_build_script_is_outside_the_kernel_graph() {
    let workspace = Workspace::new("dev-dependencies", false);
    let project = load(&workspace.options()).unwrap();
    let dependency = project
        .metadata
        .packages
        .iter()
        .find(|p| p.name == "audit-test-support")
        .unwrap();
    assert!(!project.dependencies.contains(&dependency.id));
}

#[test]
fn unused_dev_proc_macro_is_outside_the_kernel_graph() {
    let workspace = Workspace::new("dev-dependencies", true);
    assert!(load(&workspace.options()).is_ok());
}

#[test]
fn required_build_scripts_are_rejected_before_execution() {
    for kind in ["dependencies", "build-dependencies"] {
        let workspace = Workspace::new(kind, false);
        let error = load(&workspace.options()).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("kernel build scripts are unsupported")
        );
    }
}

#[test]
fn required_proc_macros_are_rejected_before_execution() {
    let workspace = Workspace::new("dependencies", true);
    let error = load(&workspace.options()).err().unwrap();
    assert!(error.to_string().contains("kernel proc macros"));
}
