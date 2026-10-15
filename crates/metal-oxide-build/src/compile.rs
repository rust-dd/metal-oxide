use crate::{
    Compiler, cache,
    metadata::{Metadata, Project},
    process::{self, Result},
    recorded::{Recorded, Request},
    rust::{KernelBuild, Rust},
    state::State,
    toolchain::{Metal, debug_flags, metal_flags},
};
use metal_oxide_artifact::{
    Abi, ArtifactFile, BuildInfo, COMPILER_NIGHTLY, COMPILER_OUTPUTS, DEVICE_TARGET, Files,
    MSL_VERSION, Manifest, MathMode, sha256,
};
use std::{
    fs::{File, OpenOptions, TryLockError},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

/// A kernel crate that the kernel macros build for Metal inside rustc, without the host
/// build's settings and with diagnostics in the error.
pub(crate) struct Kernels {
    project: Project,
    compiler: Compiler,
    math_mode: MathMode,
    /// The host crate that the kernels were copied from, which resolves their packages.
    host: Option<Arc<Metadata>>,
    /// The directory that relative source paths in line tables start at.
    debug_info: Option<PathBuf>,
}

/// The artifact that the kernel macros embed.
pub struct BuiltKernels {
    pub directory: PathBuf,
    pub abi: Abi,
    /// Files whose changes call for rebuilding the crate that embeds the artifact.
    pub tracked: Vec<PathBuf>,
    /// The artifact is the one saved beside the package's sources, which holds only the
    /// manifest and the library.
    pub saved: bool,
}

struct Compiled {
    directory: PathBuf,
    kernels: KernelBuild,
    compiler: PathBuf,
}

impl Kernels {
    /// The kernels of a kernel crate, or with `host` those copied from a host crate into a
    /// device crate.
    pub(crate) fn new(
        project: Project,
        compiler: Compiler,
        math_mode: MathMode,
        host: Option<Arc<Metadata>>,
        debug_info: Option<PathBuf>,
    ) -> Self {
        Self {
            project,
            compiler,
            math_mode,
            host,
            debug_info,
        }
    }

    /// Builds the kernels and records what they were built from, relative to `base`, in
    /// `state`, so unchanged inputs reuse the artifact.
    pub(crate) fn record(
        &self,
        base: &Path,
        request: Request,
        state: &Path,
        lock: &BuildLock,
    ) -> Result<(BuiltKernels, Recorded)> {
        let compiled = self.compiled(lock)?;
        let sysroot = process::capture(Command::new("rustup").args([
            "run",
            COMPILER_NIGHTLY,
            "rustc",
            "--print",
            "sysroot",
        ]))?;
        let inputs = Recorded::from_build(
            &self.project,
            self.host.as_deref(),
            &compiled.kernels.messages,
            Path::new(&sysroot),
            base,
            &compiled.directory.file_name().unwrap().to_string_lossy(),
            request,
        )?;
        State::new(inputs.clone(), &compiled.directory, &compiled.compiler)?.write(state)?;
        let abi = std::fs::read_to_string(compiled.directory.join(ArtifactFile::Abi.name()))?;
        let built = BuiltKernels {
            abi: Abi::from_json(&abi)?,
            directory: compiled.directory,
            tracked: std::iter::once(compiled.compiler)
                .chain(inputs.paths(base))
                .collect(),
            saved: false,
        };
        Ok((built, inputs))
    }

    fn compiled(&self, _lock: &BuildLock) -> Result<Compiled> {
        let root = self.root()?;
        cache::discard_abandoned(&root)?;
        let rust = self.rust()?;
        let kernels = rust.kernels(&self.project)?;
        let output = &kernels.output;
        let metal = Metal::find()?;
        let generated = COMPILER_OUTPUTS
            .map(|file| std::fs::read(output.join(file.name())))
            .into_iter()
            .collect::<std::io::Result<Vec<_>>>()?;
        let abi = Abi::from_json(&std::fs::read_to_string(
            output.join(ArtifactFile::Abi.name()),
        )?)?;
        let mut flags = metal_flags(&abi);
        if let Some(root) = &self.debug_info {
            flags.extend(debug_flags(root));
        }
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
            &flags,
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
            let result = create(&stage, &generated, &rust, &metal, &fingerprint, flags)
                .and_then(|()| cache::settle(&stage));
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
        }
        Ok(Compiled {
            directory,
            kernels,
            compiler: rust.compiler,
        })
    }

    fn rust(&self) -> Result<Rust> {
        Rust::prepare(
            &self.project,
            self.math_mode,
            &self.compiler,
            self.host.is_some(),
        )
    }

    fn root(&self) -> Result<PathBuf> {
        let root = self.project.metadata.target_directory.join("metal");
        std::fs::create_dir_all(&root)?;
        Ok(root)
    }
}

/// The lock on a target directory's Metal builds, which one process at a time holds.
pub(crate) struct BuildLock {
    _file: File,
}

impl BuildLock {
    pub(crate) fn acquire(target_directory: &Path) -> Result<Self> {
        let root = target_directory.join("metal");
        std::fs::create_dir_all(&root)?;
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
        Ok(Self { _file: lock })
    }
}

fn create(
    stage: &Path,
    generated: &[Vec<u8>],
    rust: &Rust,
    metal: &Metal,
    fingerprint: &str,
    flags: Vec<String>,
) -> Result<()> {
    for (file, bytes) in COMPILER_OUTPUTS.into_iter().zip(generated) {
        std::fs::write(stage.join(file.name()), bytes)?;
    }
    let abi = Abi::from_json(&std::fs::read_to_string(
        stage.join(ArtifactFile::Abi.name()),
    )?)?;
    metal.compile(stage, &flags)?;
    let manifest = Manifest {
        abi,
        checks: serde_json::from_slice(&std::fs::read(stage.join(ArtifactFile::Checks.name()))?)?,
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
