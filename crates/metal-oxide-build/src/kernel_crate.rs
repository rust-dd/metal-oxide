use crate::{
    BuiltKernels, Compiler, Error,
    compile::{BuildLock, Kernels},
    metadata, modules,
    process::Result,
    recorded::Request,
    saved::{self, Saved},
    source::host,
    state::State,
    toolchain::{self, Metal},
};
use metal_oxide_artifact::{MathMode, sha256};
use std::path::PathBuf;

/// A kernel crate that `#[kernels(path = ...)]` builds for a host crate, such as a kernel code
/// base spread over several files.
pub struct KernelCrate {
    /// The host package's manifest directory.
    pub package: PathBuf,
    /// The source file that names the kernel crate, as rustc names it, with the 1-based line and
    /// column where the module item starts; rust-analyzer gives none.
    pub location: Option<(PathBuf, (usize, usize))>,
    /// The module that launches the crate's kernels.
    pub item: String,
    /// The kernel crate's path, relative to the host package.
    pub path: PathBuf,
    pub math_mode: MathMode,
    /// Records line tables and sources for a GPU capture.
    pub debug_info: bool,
}

impl KernelCrate {
    /// Builds the crate, reuses the artifact of identical inputs without Cargo or uses the
    /// one saved beside the host package's sources.
    pub fn build(&self) -> std::result::Result<BuiltKernels, Error> {
        self.run().map_err(Error)
    }

    /// The artifact of the last build or the saved one, whatever changed since, as
    /// rust-analyzer needs it.
    pub fn previous(&self) -> Option<BuiltKernels> {
        let state = self.state().ok().and_then(|state| State::read(&state));
        if let Some(built) = state.and_then(|state| state.artifact(&self.package)) {
            return Some(built);
        }
        let (saved, request) = self.saved().ok()?;
        saved.load(&request, None).ok()
    }

    fn manifest(&self) -> PathBuf {
        self.package.join(&self.path).join("Cargo.toml")
    }

    /// The directory that records the crate's builds for this math mode and debug setting.
    fn state(&self) -> Result<PathBuf> {
        let manifest = self
            .manifest()
            .canonicalize()
            .map_err(|error| format!("kernel crate {}: {error}", self.manifest().display()))?;
        let identity = serde_json::to_vec(&(manifest, self.math_mode, self.debug_info))?;
        let workspace = host(&self.package, false)?;
        Ok(workspace
            .target_directory
            .join("metal/crates")
            .join(&sha256(&identity)[..16]))
    }

    fn saved(&self) -> Result<(Saved, Request)> {
        let (file, start) = self
            .location
            .as_ref()
            .ok_or("kernels must be launched from a source file")?;
        let path = std::env::current_dir()?.join(file);
        let file = path
            .strip_prefix(&self.package)
            .map_err(|_| "kernels must be launched from their package's directory")?;
        let modules = modules::enclosing(&std::fs::read_to_string(&path)?, *start);
        let source = serde_json::to_vec(&(file, &self.item, &self.path))?;
        let request = Request {
            source: sha256(&source),
            math_mode: self.math_mode,
            debug_info: false,
        };
        Ok((
            Saved::new(&self.package, file, &modules, &self.item, self.math_mode),
            request,
        ))
    }

    fn run(&self) -> Result<BuiltKernels> {
        let (saved, request) = self.saved()?;
        let request = Request {
            debug_info: self.debug_info,
            ..request
        };
        saved::obtain(
            &self.package,
            &saved,
            &request,
            || {
                let Ok(metal) = Metal::current() else {
                    return Ok(None);
                };
                let state = State::read(&self.state()?);
                Ok(
                    state
                        .and_then(|state| state.reuse(&self.package, &request, &metal.toolchain())),
                )
            },
            || {
                toolchain::nightly()?;
                let metal = Metal::current()?;
                let project = metadata::load(&self.manifest())?;
                let compiler = Compiler::search(&project.metadata)?;
                Ok((project, compiler, metal))
            },
            |(project, compiler, metal)| {
                let state = self.state()?;
                std::fs::create_dir_all(&state)?;
                let debug_info = self
                    .debug_info
                    .then(|| project.metadata.workspace_root.clone());
                let lock = BuildLock::acquire(&project.metadata.target_directory)?;
                Kernels::new(project, compiler, metal, self.math_mode, None, debug_info).record(
                    &self.package,
                    request.clone(),
                    &state,
                    &lock,
                )
            },
        )
    }
}
