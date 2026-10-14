#![allow(dead_code)]

use std::{
    collections::BTreeMap,
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
            "target/kernel-tests/{}-{}",
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

    /// Adds the kernel crate `kernels`, with `lib` as its root, to the project's workspace.
    pub fn kernel_crate(&self, lib: &str) {
        std::fs::create_dir_all(self.0.join("kernels/src")).unwrap();
        let device = root().join("crates/metal-oxide-device");
        std::fs::write(
            self.0.join("kernels/Cargo.toml"),
            format!("[package]\nname = \"kernels\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\nmetal-oxide-device = {{ path = {:?} }}\n", device.display().to_string()),
        )
        .unwrap();
        std::fs::write(self.0.join("kernels/src/lib.rs"), lib).unwrap();
        let manifest = self.0.join("Cargo.toml");
        let text = std::fs::read_to_string(&manifest).unwrap();
        let text = text.replace("[workspace]\n", "[workspace]\nmembers = [\"kernels\"]\n");
        std::fs::write(manifest, text).unwrap();
        checked(
            Command::new("cargo")
                .current_dir(&self.0)
                .args(["generate-lockfile", "--offline"])
                .output()
                .unwrap(),
        );
    }

    /// Runs a Cargo `command` and returns the artifacts that its kernel builds compiled, by item.
    pub fn built(&self, command: &mut Command) -> BTreeMap<String, PathBuf> {
        let list = self.0.join("inspected");
        let _ = std::fs::remove_file(&list);
        checked(command.env("METAL_OXIDE_INSPECT", &list).output().unwrap());
        std::fs::read_to_string(&list)
            .unwrap_or_default()
            .lines()
            .map(|line| {
                let [directory, _, item] = line.split('\t').collect::<Vec<_>>()[..] else {
                    panic!("{line}");
                };
                (item.to_owned(), PathBuf::from(directory))
            })
            .collect()
    }

    /// The target directory that projects share, so each builds the runtime once.
    pub fn target() -> PathBuf {
        root().join("target/kernel-tests/target")
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
    /// debug setting recorded.
    pub fn artifact_of(
        &self,
        item: &str,
        math_mode: metal_oxide_artifact::MathMode,
        debug_info: bool,
    ) -> PathBuf {
        let state = self.state(item, math_mode, debug_info);
        PathBuf::from(state["artifact"].as_str().unwrap())
    }

    /// The reuse state of the device crate that metal-oxide-build keeps for `item` in
    /// `src/main.rs`.
    pub fn state(
        &self,
        item: &str,
        math_mode: metal_oxide_artifact::MathMode,
        debug_info: bool,
    ) -> serde_json::Value {
        let identity =
            serde_json::to_vec(&(&self.0, "src/main.rs", item, math_mode, debug_info)).unwrap();
        let key = &metal_oxide_artifact::sha256(&identity)[..16];
        let state = Self::target()
            .join("metal/sources")
            .join(key)
            .join("state.json");
        serde_json::from_slice(&std::fs::read(&state).unwrap()).unwrap()
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

/// The errors of a command that must fail.
pub fn failure(output: Output) -> String {
    assert!(!output.status.success());
    String::from_utf8_lossy(&output.stderr).into_owned()
}
