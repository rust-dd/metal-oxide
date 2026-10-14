#![allow(dead_code)]

use std::{
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// A package that depends on this checkout's metal-oxide; it is removed when dropped.
pub struct Project(pub PathBuf);

impl Project {
    /// A package whose `src/main.rs` is `main`; `{root}` in it names this checkout.
    pub fn new(main: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = root().join(format!(
            "target/cli-tests/{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(path.join("src")).unwrap();
        let runtime = root().join("crates/metal-oxide");
        std::fs::write(
            path.join("Cargo.toml"),
            format!("[package]\nname = \"host\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\nmetal-oxide = {{ path = {:?} }}\n\n[workspace]\n", runtime.display().to_string()),
        )
        .unwrap();
        let project = Self(path);
        project.write(main);
        checked(
            Command::new("cargo")
                .current_dir(&project.0)
                .args(["generate-lockfile", "--offline"])
                .output()
                .unwrap(),
        );
        project
    }

    pub fn write(&self, main: &str) {
        let main = main.replace("{root}", &root().display().to_string());
        std::fs::write(self.0.join("src/main.rs"), main).unwrap();
    }

    /// The target directory that projects share, so each builds the runtime once.
    pub fn target() -> PathBuf {
        root().join("target/kernel-tests/target")
    }

    /// `cargo metal` in the project, with the target directory that projects share.
    pub fn metal(&self, arguments: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_cargo-metal"));
        command
            .current_dir(&self.0)
            .arg("metal")
            .args(arguments)
            .env("CARGO_TARGET_DIR", Self::target())
            .env_remove("RUST_ANALYZER_INTERNALS_DO_NOT_USE");
        command
    }

    pub fn cargo(&self, arguments: &[&str]) -> Command {
        let mut command = Command::new("cargo");
        command
            .current_dir(&self.0)
            .args(arguments)
            .env("CARGO_TARGET_DIR", Self::target())
            .env_remove("RUST_ANALYZER_INTERNALS_DO_NOT_USE");
        command
    }

    /// The artifact that the last build of `item` in `src/main.rs` recorded.
    pub fn artifact(&self, item: &str) -> PathBuf {
        self.artifact_of(item, metal_oxide_artifact::MathMode::Precise, false)
    }

    /// The artifact that the last build of `item` in `src/main.rs` with this math mode and
    /// debug setting recorded, as metal-oxide-build names its device crate.
    pub fn artifact_of(
        &self,
        item: &str,
        math_mode: metal_oxide_artifact::MathMode,
        debug_info: bool,
    ) -> PathBuf {
        let identity =
            serde_json::to_vec(&(&self.0, "src/main.rs", item, math_mode, debug_info)).unwrap();
        let key = &metal_oxide_artifact::sha256(&identity)[..16];
        let state = Self::target()
            .join("metal/sources")
            .join(key)
            .join("state.json");
        let state: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&state).unwrap()).unwrap();
        PathBuf::from(state["artifact"].as_str().unwrap())
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
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
