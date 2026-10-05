use crate::{
    metadata::Project,
    process::{self, Result},
};
use metal_oxide_artifact::sha256;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

pub(crate) const NIGHTLY: &str = "nightly-2026-10-04";
const TARGET: &str = include_str!("../../../targets/metal64-unknown-none.json");

pub(crate) struct Rust {
    pub(crate) compiler: PathBuf,
    pub(crate) identity: String,
    pub(crate) version: String,
    pub(crate) target: PathBuf,
    pub(crate) flags: Vec<String>,
    pub(crate) directory: PathBuf,
}

fn cargo() -> Command {
    let mut command = Command::new("rustup");
    command.args(["run", NIGHTLY, "cargo"]);
    command
}

impl Rust {
    pub(crate) fn prepare(project: &Project) -> Result<Self> {
        let root = project.metadata.target_directory.join("metal");
        std::fs::create_dir_all(&root)?;
        let target = root.join("metal64-unknown-none.json");
        if std::fs::read_to_string(&target).ok().as_deref() != Some(TARGET) {
            std::fs::write(&target, TARGET)?;
        }
        let target = target.canonicalize()?;
        let compiler = if let Some(path) = std::env::var_os("METAL_OXIDE_COMPILER") {
            PathBuf::from(path).canonicalize()?
        } else {
            let source =
                Path::new(env!("CARGO_MANIFEST_DIR")).join("../metal-oxide-compiler/Cargo.toml");
            let directory = project.metadata.target_directory.join("compiler");
            process::run(
                cargo()
                    .args([
                        "build",
                        "--features",
                        "rustc-private",
                        "--bin",
                        "metal-oxide-compiler",
                        "--locked",
                    ])
                    .arg("--manifest-path")
                    .arg(source)
                    .arg("--target-dir")
                    .arg(&directory),
            )?;
            directory
                .join("debug/metal-oxide-compiler")
                .canonicalize()?
        };
        let version =
            process::capture(Command::new("rustup").args(["run", NIGHTLY, "rustc", "-vV"]))?;
        let identity = sha256(&std::fs::read(&compiler)?);
        let mut flags = if let Ok(value) = std::env::var("CARGO_ENCODED_RUSTFLAGS") {
            value
                .split('\u{1f}')
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        } else {
            std::env::var("RUSTFLAGS")
                .unwrap_or_default()
                .split_whitespace()
                .map(str::to_owned)
                .collect()
        };
        flags.push("-Zunstable-options".into());
        let configuration = serde_json::to_vec(&(&identity, &version, TARGET, &flags))?;
        let directory = root.join("rust").join(sha256(&configuration));
        Ok(Self {
            compiler,
            identity,
            version,
            target,
            flags,
            directory,
        })
    }

    fn command(&self, project: &Project, output: &Path) -> Command {
        let mut command = cargo();
        command
            .current_dir(&project.metadata.workspace_root)
            .env("RUSTC_WRAPPER", &self.compiler)
            .env_remove("RUSTC_WORKSPACE_WRAPPER")
            .env_remove("RUSTC")
            .env_remove("RUSTFLAGS")
            .env("CARGO_ENCODED_RUSTFLAGS", self.flags.join("\u{1f}"))
            .env("METAL_OXIDE_CARGO_TARGET", &self.target)
            .env(
                "METAL_OXIDE_CARGO_KERNEL",
                project.kernel_crate().replace('-', "_"),
            )
            .env("METAL_OXIDE_CARGO_OUTPUT", output);
        command
    }

    pub(crate) fn kernels(&self, project: &Project) -> Result<PathBuf> {
        let output = self
            .directory
            .join(format!("output-{}", sha256(project.kernel().id.as_bytes())));
        std::fs::create_dir_all(&output)?;
        self.check(project, &output)?;
        if [
            "kernels.metal",
            "kernels.oxide-ir",
            "abi.json",
            "bindings.rs",
        ]
        .iter()
        .any(|name| !output.join(name).is_file())
        {
            process::run(
                self.command(project, &output)
                    .args([
                        "clean",
                        "-Zjson-target-spec",
                        "--release",
                        "--locked",
                        "--package",
                        &project.kernel().name,
                    ])
                    .arg("--manifest-path")
                    .arg(&project.kernel().manifest_path)
                    .arg("--target")
                    .arg(&self.target)
                    .arg("--target-dir")
                    .arg(&self.directory),
            )?;
            self.check(project, &output)?;
        }
        Ok(output)
    }

    fn check(&self, project: &Project, output: &Path) -> Result<()> {
        process::run(
            self.command(project, output)
                .args([
                    "check",
                    "-Zbuild-std=core",
                    "-Zjson-target-spec",
                    "--lib",
                    "--release",
                    "--locked",
                    "--package",
                    &project.kernel().name,
                ])
                .arg("--manifest-path")
                .arg(&project.kernel().manifest_path)
                .arg("--target")
                .arg(&self.target)
                .arg("--target-dir")
                .arg(&self.directory),
        )
    }
}
