use crate::{
    Compiler, Error, cache, metadata,
    metadata::Project,
    process::Result,
    rust::{KernelBuild, Rust},
    toolchain::{Metal, metal_flags},
};
use metal_oxide_artifact::{
    Abi, ArtifactFile, BuildInfo, COMPILER_OUTPUTS, DEVICE_TARGET, Files, MSL_VERSION, Manifest,
    MathMode, sha256,
};
use std::{
    fs::{File, OpenOptions, TryLockError},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

/// A kernel crate that the metal-oxide compiler builds for Metal.
pub struct Kernels {
    project: Project,
    compiler: Compiler,
    math_mode: MathMode,
    isolated: bool,
}

pub(crate) struct Compiled {
    pub(crate) directory: PathBuf,
    pub(crate) kernels: KernelBuild,
}

impl Kernels {
    /// Reads the Cargo workspace of the kernel crate whose Cargo.toml is `manifest`.
    pub fn new(
        manifest: impl AsRef<Path>,
        compiler: Compiler,
        math_mode: MathMode,
    ) -> std::result::Result<Self, Error> {
        let project = metadata::load(manifest.as_ref()).map_err(Error)?;
        Ok(Self {
            project,
            compiler,
            math_mode,
            isolated: false,
        })
    }

    /// Kernels built from a build script, without the host build's Rust flags.
    pub(crate) fn from_build_script(
        project: Project,
        compiler: Compiler,
        math_mode: MathMode,
    ) -> Self {
        Self {
            project,
            compiler,
            math_mode,
            isolated: true,
        }
    }

    pub(crate) fn project(&self) -> &Project {
        &self.project
    }

    /// The generated MSL; Apple's Metal toolchain is not needed.
    pub fn msl(&self) -> std::result::Result<String, Error> {
        let msl = || -> Result<String> {
            let root = self.root()?;
            let _lock = lock(&root)?;
            let rust = Rust::prepare(&self.project, self.math_mode, &self.compiler, self.isolated)?;
            let kernels = rust.kernels(&self.project)?;
            Ok(std::fs::read_to_string(
                kernels.output.join(ArtifactFile::Msl.name()),
            )?)
        };
        msl().map_err(Error)
    }

    /// Builds the artifact under the Cargo target directory, or reuses the build with
    /// identical inputs and toolchain, and returns its directory.
    pub fn compile(&self) -> std::result::Result<PathBuf, Error> {
        self.compiled()
            .map(|compiled| compiled.directory)
            .map_err(Error)
    }

    pub(crate) fn compiled(&self) -> Result<Compiled> {
        let root = self.root()?;
        let lock = lock(&root)?;
        cache::discard_abandoned(&root)?;
        let rust = Rust::prepare(&self.project, self.math_mode, &self.compiler, self.isolated)?;
        let kernels = rust.kernels(&self.project)?;
        let output = &kernels.output;
        let metal = Metal::find()?;
        let generated = COMPILER_OUTPUTS
            .map(|file| std::fs::read(output.join(file.name())))
            .into_iter()
            .collect::<std::io::Result<Vec<_>>>()?;
        let profile = std::env::vars()
            .filter(|(key, _)| key.starts_with("CARGO_PROFILE_RELEASE_"))
            .collect::<std::collections::BTreeMap<_, _>>();
        let identity = serde_json::to_vec(&(
            env!("CARGO_PKG_VERSION"),
            &kernels.inputs,
            &generated,
            &rust.identity,
            &rust.version,
            std::fs::read(&rust.target)?,
            &metal.version,
            &metal.sdk,
            metal_flags(self.math_mode),
            profile,
        ))?;
        let fingerprint = sha256(&identity);
        let directory = root.join(&fingerprint);
        if !cache::valid(&directory, &fingerprint) {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let stage = root.join(format!(
                ".build-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&stage)?;
            let result = create(&stage, &generated, &rust, &metal, &fingerprint);
            if let Err(error) = result {
                let failed = root.join(
                    stage
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .replacen(".build-", ".failed-", 1),
                );
                std::fs::rename(&stage, &failed)?;
                return Err(format!(
                    "Metal build failed; inputs retained at {}: {error}",
                    failed.display()
                )
                .into());
            }
            if directory.exists() {
                std::fs::remove_dir_all(&directory)?;
            }
            std::fs::rename(&stage, &directory)?;
            eprintln!("Built Metal artifact {}", directory.display());
        } else {
            eprintln!("Cached Metal artifact {}", directory.display());
        }
        drop(lock);
        Ok(Compiled { directory, kernels })
    }

    fn root(&self) -> Result<PathBuf> {
        let root = self.project.metadata.target_directory.join("metal");
        std::fs::create_dir_all(&root)?;
        Ok(root)
    }
}

fn lock(root: &Path) -> Result<File> {
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(".lock"))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            eprintln!("Blocking waiting for Metal build lock");
            lock.lock()?;
        }
        Err(TryLockError::Error(error)) => return Err(error.into()),
    }
    Ok(lock)
}

fn create(
    stage: &Path,
    generated: &[Vec<u8>],
    rust: &Rust,
    metal: &Metal,
    fingerprint: &str,
) -> Result<()> {
    for (file, bytes) in COMPILER_OUTPUTS.into_iter().zip(generated) {
        std::fs::write(stage.join(file.name()), bytes)?;
    }
    let abi = Abi::from_json(&std::fs::read_to_string(
        stage.join(ArtifactFile::Abi.name()),
    )?)?;
    metal.compile(stage, abi.math_mode)?;
    let flags = metal_flags(abi.math_mode);
    let manifest = Manifest {
        abi,
        target: DEVICE_TARGET.into(),
        msl_version: MSL_VERSION.into(),
        files: Files::from_directory(stage)?,
        build: BuildInfo {
            fingerprint: fingerprint.into(),
            compiler: rust.identity.clone(),
            rustc: rust.version.clone(),
            metal: metal.version.clone(),
            sdk: metal.sdk.clone(),
            rustc_args: serde_json::from_slice(&std::fs::read(
                stage.join(ArtifactFile::RustcArgs.name()),
            )?)?,
            metal_flags: flags,
        },
    };
    std::fs::write(
        stage.join(ArtifactFile::Manifest.name()),
        manifest.to_json()?,
    )?;
    Ok(())
}
