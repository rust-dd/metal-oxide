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
    Abi, ArtifactFile, BuildInfo, COMPILER_NIGHTLY, COMPILER_OUTPUTS, Files, Manifest, MathMode,
    sha256,
};
use std::{
    collections::BTreeMap,
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
    /// Referenced ambient variables whose changes rebuild the embedding crate.
    pub environment: Vec<String>,
    /// The artifact is the one saved beside the package's sources, which holds only the
    /// manifest and the library.
    pub saved: bool,
}

struct Compiled {
    directory: PathBuf,
    abi: Abi,
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
        let built = BuiltKernels {
            abi: compiled.abi,
            directory: compiled.directory,
            tracked: std::iter::once(compiled.compiler)
                .chain(inputs.paths(base))
                .collect(),
            saved: false,
            environment: inputs.environment.names(),
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
            .into_iter()
            .map(|file| Ok((file.name(), std::fs::read(output.join(file.name()))?)))
            .collect::<std::io::Result<BTreeMap<_, _>>>()?;
        let abi = Abi::from_json(std::str::from_utf8(&generated[ArtifactFile::Abi.name()])?)?;
        let mut flags = metal_flags(&abi);
        if let Some(root) = &self.debug_info {
            flags.extend(debug_flags(root));
        }
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
            crate::recorded::profile(),
        ))?;
        let fingerprint = sha256(&identity);
        let directory = root.join(&fingerprint);
        let manifest = match cache::manifest(&directory, &fingerprint) {
            Some(manifest) => manifest,
            None => {
                let build = BuildInfo {
                    fingerprint,
                    compiler: rust.identity.clone(),
                    rustc: rust.version.clone(),
                    // metal-commands.json keeps the full text, which names the toolchain's path.
                    metal: metal.version.lines().next().unwrap_or_default().into(),
                    sdk: metal.sdk.clone(),
                    metal_flags: flags,
                };
                staged(&root, &directory, |stage| {
                    create(stage, &generated, abi, &metal, build)
                })?
            }
        };
        Ok(Compiled {
            directory,
            abi: manifest.abi,
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
            self.debug_info.is_some(),
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

/// Creates an artifact in a new stage directory that then replaces `directory`; a failed
/// build keeps its stage for inspection.
fn staged(
    root: &Path,
    directory: &Path,
    create: impl FnOnce(&Path) -> Result<Manifest>,
) -> Result<Manifest> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let stage = root.join(format!(
        ".build-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&stage)?;
    let result = create(&stage).and_then(|manifest| cache::settle(&stage).map(|()| manifest));
    let manifest = match result {
        Ok(manifest) => manifest,
        Err(error) => {
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
    };
    if directory.exists() {
        std::fs::remove_dir_all(directory)?;
    }
    std::fs::rename(&stage, directory)?;
    Ok(manifest)
}

fn create(
    stage: &Path,
    generated: &BTreeMap<&str, Vec<u8>>,
    abi: Abi,
    metal: &Metal,
    build: BuildInfo,
) -> Result<Manifest> {
    // The manifest holds the ABI and the checks; the other compiler outputs stay beside it.
    for file in [
        ArtifactFile::Msl,
        ArtifactFile::OxideIr,
        ArtifactFile::RustcArgs,
    ] {
        std::fs::write(stage.join(file.name()), &generated[file.name()])?;
    }
    metal.compile(stage, &build.metal_flags)?;
    let manifest = Manifest {
        abi,
        checks: serde_json::from_slice(&generated[ArtifactFile::Checks.name()])?,
        files: Files::from_directory(stage)?,
        build,
    };
    std::fs::write(
        stage.join(ArtifactFile::Manifest.name()),
        manifest.to_json()?,
    )?;
    Ok(manifest)
}
