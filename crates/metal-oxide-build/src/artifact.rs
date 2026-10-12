use crate::{
    Compiler, Error,
    compile::Kernels,
    metadata,
    process::{self, Result},
    recorded::Recorded,
};
use metal_oxide_artifact::{ArtifactFile, COMPILER_NIGHTLY, Manifest, MathMode, sha256};
use std::{
    path::{Path, PathBuf},
    process::Command,
};

/// The files a host embeds with `include_str!`, `include_bytes!` and `include!`.
const EMBEDDED: [ArtifactFile; 3] = [
    ArtifactFile::Manifest,
    ArtifactFile::Metallib,
    ArtifactFile::Bindings,
];
const INPUTS: &str = "inputs.json";

/// A kernel artifact kept in the calling package's sources, so the package builds and
/// publishes without the kernel toolchain. Call [`Artifact::update`] from its build script.
pub struct Artifact {
    kernels: PathBuf,
    directory: PathBuf,
    math_mode: MathMode,
}

impl Artifact {
    /// `kernels` is the kernel crate's Cargo.toml and `directory` holds the artifact files;
    /// relative paths start at the calling package.
    pub fn new(kernels: impl Into<PathBuf>, directory: impl Into<PathBuf>) -> Self {
        Self {
            kernels: kernels.into(),
            directory: directory.into(),
            math_mode: MathMode::default(),
        }
    }

    pub fn math_mode(mut self, math_mode: MathMode) -> Self {
        self.math_mode = math_mode;
        self
    }

    /// Keeps the artifact while the kernel inputs recorded beside it are unchanged and
    /// rebuilds it otherwise. Without the kernel sources, as in a published package, the
    /// artifact is used as is. With `METAL_OXIDE_ARTIFACTS=locked` a stale artifact is an error.
    pub fn update(self) -> std::result::Result<(), Error> {
        self.run().map_err(Error)
    }

    fn run(self) -> Result<()> {
        println!("cargo:rerun-if-env-changed=METAL_OXIDE_ARTIFACTS");
        println!("cargo:rerun-if-env-changed=METAL_OXIDE_COMPILER");
        let package = PathBuf::from(
            std::env::var_os("CARGO_MANIFEST_DIR")
                .ok_or("Artifact::update runs in a build script")?,
        );
        let kernels = package.join(&self.kernels);
        let directory = package.join(&self.directory);
        if !kernels.is_file() {
            for file in EMBEDDED.map(ArtifactFile::name) {
                if !directory.join(file).is_file() {
                    return Err(format!(
                        "{} has no {file} and the kernel crate {} is absent",
                        directory.display(),
                        kernels.display()
                    )
                    .into());
                }
            }
            return Ok(());
        }
        let base = kernels.parent().unwrap();
        if let Some(recorded) = current(&directory, self.math_mode)
            && recorded.current(base)
        {
            rerun(&recorded, base);
            return Ok(());
        }
        let mode = std::env::var("METAL_OXIDE_ARTIFACTS");
        match mode.as_deref() {
            Ok("locked") => {
                return Err(format!(
                    "the Metal artifact in {} does not match its kernel sources; rebuild without METAL_OXIDE_ARTIFACTS=locked",
                    directory.display()
                )
                .into());
            }
            Ok(value) => {
                return Err(format!("METAL_OXIDE_ARTIFACTS must be locked, not {value:?}").into());
            }
            Err(std::env::VarError::NotPresent) => {}
            Err(error) => return Err(error.to_string().into()),
        }
        let project = metadata::load(&kernels)?;
        let compiler = Compiler::search(&project)?;
        let build = Kernels::from_build_script(project, compiler, self.math_mode);
        let compiled = build.compiled()?;
        let manifest = Manifest::from_json(&std::fs::read_to_string(
            compiled.directory.join(ArtifactFile::Manifest.name()),
        )?)?;
        let sysroot = process::capture(Command::new("rustup").args([
            "run",
            COMPILER_NIGHTLY,
            "rustc",
            "--print",
            "sysroot",
        ]))?;
        let recorded = Recorded::from_build(
            build.project(),
            &compiled.kernels.messages,
            Path::new(&sysroot),
            &manifest.build.fingerprint,
            self.math_mode,
        )?;
        std::fs::create_dir_all(&directory)?;
        for file in EMBEDDED.map(ArtifactFile::name) {
            replace(
                &directory.join(file),
                &std::fs::read(compiled.directory.join(file))?,
            )?;
        }
        replace(
            &directory.join(INPUTS),
            (serde_json::to_string_pretty(&recorded)? + "\n").as_bytes(),
        )?;
        // Cargo replays build script warnings on fresh builds, so this goes to `-vv` output.
        eprintln!(
            "rebuilt the Metal artifact in {} from changed kernel inputs",
            directory.display()
        );
        rerun(&recorded, base);
        Ok(())
    }
}

/// The recorded inputs of an intact artifact built in `math_mode`.
fn current(directory: &Path, math_mode: MathMode) -> Option<Recorded> {
    let read = |file: &str| std::fs::read(directory.join(file)).ok();
    let recorded: Recorded = serde_json::from_slice(&read(INPUTS)?).ok()?;
    let manifest =
        Manifest::from_json(&String::from_utf8(read(ArtifactFile::Manifest.name())?).ok()?).ok()?;
    let intact = manifest.build.fingerprint == recorded.artifact
        && recorded.math_mode == math_mode
        && manifest.abi.math_mode == math_mode
        && manifest
            .verify_library(&read(ArtifactFile::Metallib.name())?)
            .is_ok()
        && sha256(&read(ArtifactFile::Bindings.name())?) == manifest.files.bindings;
    intact.then_some(recorded)
}

fn rerun(recorded: &Recorded, base: &Path) {
    for path in recorded.paths(base) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// Writes beside `path` first so readers never see a partial file.
fn replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let partial = path.with_file_name(format!(
        ".{}.partial",
        path.file_name().unwrap().to_string_lossy()
    ));
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, path)?;
    Ok(())
}
