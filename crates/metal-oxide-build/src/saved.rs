use crate::{
    BuiltKernels,
    process::Result,
    recorded::{Recorded, Request},
};
use metal_oxide_artifact::{ArtifactFile, Files, Manifest, MathMode};
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

const INPUTS: &str = "inputs.json";

/// An artifact that a package keeps in `metal/<file>/<modules>/<item>-<math mode>` beside its
/// sources, so that it builds without the kernel toolchain, as for its dependents.
pub(crate) struct Saved {
    directory: PathBuf,
    /// The directory relative to the package, for messages.
    name: PathBuf,
}

impl Saved {
    /// The artifact of `item` in the inline `modules` of `file`, relative to `package`.
    pub(crate) fn new(
        package: &Path,
        file: &Path,
        modules: &[String],
        item: &str,
        math_mode: MathMode,
    ) -> Self {
        let mut name = Path::new("metal").join(file);
        name.extend(modules);
        name.push(format!("{item}-{}", math_mode.as_str()));
        Self {
            directory: package.join(&name),
            name,
        }
    }

    /// The saved artifact, if it is intact and was built for `request`; with a `base`, its
    /// recorded inputs must also be unchanged relative to it.
    pub(crate) fn load(&self, request: &Request, base: Option<&Path>) -> Result<BuiltKernels> {
        let directory = self.name.display();
        let (inputs, manifest) = self.read().map_err(|error| match error {
            Read::Missing => format!("{directory} holds no saved Metal artifact"),
            Read::Damaged(reason) => {
                format!("the saved Metal artifact in {directory} is damaged: {reason}")
            }
        })?;
        if inputs.request != *request || base.is_some_and(|base| !inputs.current(base)) {
            return Err(format!("the saved Metal artifact in {directory} is stale").into());
        }
        Ok(BuiltKernels {
            directory: self.directory.clone(),
            abi: manifest.abi,
            tracked: base.map(|base| inputs.paths(base)).unwrap_or_default(),
            saved: true,
            environment: base.map(|_| inputs.environment.names()).unwrap_or_default(),
        })
    }

    /// Whether the saved artifact holds `built`'s kernels and was recorded from `inputs`. A build
    /// identity alone, such as that of another compiler build or machine, changes nothing.
    fn holds(&self, built: &BuiltKernels, inputs: &Recorded) -> Result<bool> {
        let Ok((saved_inputs, saved)) = self.read() else {
            return Ok(false);
        };
        let manifest = built.directory.join(ArtifactFile::Manifest.name());
        let built = Manifest::from_json(&std::fs::read_to_string(manifest)?)?;
        // The rustc arguments name the building machine's paths.
        let files = Files {
            rustc_args: saved.files.rustc_args.clone(),
            ..built.files
        };
        let inputs = Recorded {
            artifact: saved_inputs.artifact.clone(),
            ..inputs.clone()
        };
        Ok(saved.abi == built.abi
            && saved.checks == built.checks
            && saved.files == files
            && saved_inputs == inputs)
    }

    fn read(&self) -> std::result::Result<(Recorded, Manifest), Read> {
        let read = |file: &str| match std::fs::read(self.directory.join(file)) {
            Ok(bytes) => Ok(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Err(Read::Missing),
            Err(error) => Err(Read::Damaged(error.to_string())),
        };
        let inputs = read(INPUTS)?;
        let manifest = read(ArtifactFile::Manifest.name())?;
        let library = read(ArtifactFile::Metallib.name())?;
        let damaged = |error: &dyn std::fmt::Display| Read::Damaged(error.to_string());
        let inputs =
            serde_json::from_slice::<serde_json::Value>(&inputs).map_err(|e| damaged(&e))?;
        if inputs["format"] != crate::recorded::FORMAT {
            return Err(Read::Damaged(format!(
                "unsupported saved input format {}",
                inputs["format"]
            )));
        }
        let inputs = serde_json::from_value::<Recorded>(inputs).map_err(|e| damaged(&e))?;
        let manifest = String::from_utf8(manifest).map_err(|e| damaged(&e))?;
        let manifest = Manifest::from_json(&manifest).map_err(|e| damaged(&e))?;
        if manifest.build.fingerprint != inputs.artifact {
            return Err(Read::Damaged("its manifest and inputs differ".into()));
        }
        manifest.verify_library(&library).map_err(|e| damaged(&e))?;
        Ok((inputs, manifest))
    }

    fn write(&self, built: &BuiltKernels, inputs: &Recorded) -> Result<()> {
        std::fs::create_dir_all(&self.directory)?;
        for file in [ArtifactFile::Manifest, ArtifactFile::Metallib].map(ArtifactFile::name) {
            let bytes = std::fs::read(built.directory.join(file))?;
            replace(&self.directory.join(file), &bytes)?;
        }
        let inputs = serde_json::to_string_pretty(inputs)? + "\n";
        replace(&self.directory.join(INPUTS), inputs.as_bytes())
    }
}

enum Read {
    Missing,
    Damaged(String),
}

/// Obtains the artifact of one kernel item. Distributed packages and builds with
/// `METAL_OXIDE_ARTIFACTS=locked` use the saved artifact. Otherwise `reuse` or, with the
/// toolchain that `toolchain` finds, `build` makes the artifact and the saved one follows it;
/// without the toolchain, the saved artifact stands in while it is current.
pub(crate) fn obtain<T>(
    package: &Path,
    saved: &Saved,
    request: &Request,
    reuse: impl FnOnce() -> Result<Option<(BuiltKernels, Recorded)>>,
    toolchain: impl FnOnce() -> Result<T>,
    build: impl FnOnce(T) -> Result<(BuiltKernels, Recorded)>,
) -> Result<BuiltKernels> {
    if distributed(package) {
        let request = Request {
            debug_info: false,
            ..request.clone()
        };
        return saved.load(&request, None).map_err(|error| {
            format!("{error}; a published package carries the artifacts of its kernels").into()
        });
    }
    if locked()? {
        if request.debug_info {
            return Err("METAL_OXIDE_ARTIFACTS=locked rules out builds with line tables".into());
        }
        return saved.load(request, Some(package)).map_err(|error| {
            format!("{error}; METAL_OXIDE_ARTIFACTS=locked uses saved artifacts only").into()
        });
    }
    let (built, inputs) = match reuse()? {
        Some(reused) => reused,
        None => match toolchain() {
            Ok(toolchain) => build(toolchain)?,
            Err(missing) if !request.debug_info => {
                return saved.load(request, Some(package)).map_err(|error| {
                    format!("{error}, and building kernels needs the toolchain: {missing}").into()
                });
            }
            Err(missing) => return Err(missing),
        },
    };
    if !request.debug_info && !saved.holds(&built, &inputs)? {
        saved.write(&built, &inputs)?;
    }
    Ok(built)
}

/// Cargo keeps distributed sources intact, including archives unpacked for verification.
fn distributed(package: &Path) -> bool {
    package.join(".cargo-checksum.json").is_file()
        || package
            .ancestors()
            .any(|directory| directory.join(".cargo-ok").is_file())
        || package.join("Cargo.toml.orig").is_file()
            && std::fs::read_to_string(package.join("Cargo.toml")).is_ok_and(|manifest| {
                manifest.starts_with("# THIS FILE IS AUTOMATICALLY GENERATED BY CARGO\n")
            })
}

fn locked() -> Result<bool> {
    match std::env::var("METAL_OXIDE_ARTIFACTS") {
        Ok(value) if value == "locked" => Ok(true),
        Ok(value) => Err(format!("METAL_OXIDE_ARTIFACTS must be locked, not {value:?}").into()),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(error.into()),
    }
}

/// Writes beside `path` first so readers never see a partial file.
fn replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path.file_name().unwrap().to_string_lossy();
    let partial = path.with_file_name(format!(".{name}.{}.partial", std::process::id()));
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, path)?;
    Ok(())
}
