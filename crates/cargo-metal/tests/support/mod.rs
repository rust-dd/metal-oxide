#![allow(dead_code)]

use std::{
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct Workspace(pub PathBuf);

impl Workspace {
    pub fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let path = root.join(format!(
            "target/cli-tests/{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        for name in ["host", "kernels", "host-only"] {
            std::fs::create_dir_all(path.join(name).join("src")).unwrap();
        }
        std::fs::write(
            path.join("Cargo.toml"),
            "[workspace]\nmembers=[\"host\",\"kernels\",\"host-only\"]\nresolver=\"3\"\n",
        )
        .unwrap();
        std::fs::write(path.join("host/Cargo.toml"),
            "[package]\nname=\"test-host\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[package.metadata.metal]\nkernels=\"../kernels/Cargo.toml\"\n").unwrap();
        std::fs::write(path.join("host/src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(path.join("kernels/Cargo.toml"), format!(
            "[package]\nname=\"test-kernels\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[dependencies]\nmetal-oxide-device={{path={:?}}}\n[target.'cfg(not(target_env = \"metal\"))'.dependencies]\nhost-only={{path=\"../host-only\"}}\n",
            root.join("crates/metal-oxide-device"))).unwrap();
        std::fs::write(
            path.join("kernels/src/lib.rs"),
            "#![no_std]\nuse metal_oxide_device::kernel;\n#[kernel]\npub unsafe fn empty() {}\n",
        )
        .unwrap();
        std::fs::write(
            path.join("host-only/Cargo.toml"),
            "[package]\nname=\"host-only\"\nversion=\"0.1.0\"\nedition=\"2024\"\n",
        )
        .unwrap();
        std::fs::write(path.join("host-only/src/lib.rs"), "#![no_std]\n").unwrap();
        std::fs::write(
            path.join("host-only/build.rs"),
            "fn main() { panic!(\"host-only build script must not execute\") }\n",
        )
        .unwrap();
        checked(
            Command::new("cargo")
                .current_dir(&path)
                .args(["generate-lockfile", "--offline"])
                .output()
                .unwrap(),
        );
        Self(path)
    }

    pub fn command(&self, action: &str) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cargo-metal"));
        command
            .current_dir(&self.0)
            .env("CARGO_TARGET_DIR", self.0.join("target"))
            .args([action, "-p", "test-host"]);
        command
    }

    pub fn compiler(&self) -> PathBuf {
        let version = checked(
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
        let version = String::from_utf8(version.stdout).unwrap();
        let host = version
            .lines()
            .find_map(|line| line.strip_prefix("host: "))
            .unwrap();
        self.0
            .join("target/compiler")
            .join(host)
            .join("debug/metal-oxide-compiler")
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

pub fn checked(output: Output) -> Output {
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
