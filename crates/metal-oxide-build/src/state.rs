use crate::{
    BuiltKernels, cache,
    process::Result,
    recorded::{Recorded, Request, dependencies},
};
use metal_oxide_artifact::{Abi, ArtifactFile};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::SystemTime,
};

const FILE: &str = "state.json";

fn requested() -> Option<PathBuf> {
    std::env::var_os("METAL_OXIDE_COMPILER").map(PathBuf::from)
}

/// The last build of a kernel package, which an unchanged package reuses without Cargo or the
/// toolchain.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct State {
    inputs: Recorded,
    artifact: PathBuf,
    compiler: PathBuf,
    /// `METAL_OXIDE_COMPILER` when the compiler was chosen.
    requested: Option<PathBuf>,
    /// The compiler and, when it was built from source, the files it was built from.
    stamps: BTreeMap<PathBuf, Stamp>,
}

#[derive(PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stamp {
    length: u64,
    modified: SystemTime,
}

impl Stamp {
    fn of(path: &Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        Some(Self {
            length: metadata.len(),
            modified: metadata.modified().ok()?,
        })
    }
}

impl State {
    pub(crate) fn new(inputs: Recorded, artifact: &Path, compiler: &Path) -> Result<Self> {
        let mut paths = vec![compiler.to_owned()];
        // Cargo writes the sources of a compiler it built beside the executable.
        if let Ok(info) = std::fs::read_to_string(compiler.with_extension("d")) {
            paths.extend(dependencies(&info, compiler.parent().unwrap()));
        }
        let stamps = paths
            .into_iter()
            .map(|path| match Stamp::of(&path) {
                Some(stamp) => Ok((path, stamp)),
                None => Err(format!("cannot read {}", path.display()).into()),
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            inputs,
            artifact: artifact.to_owned(),
            compiler: compiler.to_owned(),
            requested: requested(),
            stamps,
        })
    }

    pub(crate) fn read(directory: &Path) -> Option<Self> {
        serde_json::from_slice(&std::fs::read(directory.join(FILE)).ok()?).ok()
    }

    /// Replaces the recorded state; a concurrent reader sees the old or the new file.
    pub(crate) fn write(&self, directory: &Path) -> Result<()> {
        let partial = directory.join(format!(".{FILE}.{}", std::process::id()));
        std::fs::write(&partial, serde_json::to_vec(self)?)?;
        std::fs::rename(&partial, directory.join(FILE))?;
        Ok(())
    }

    /// The recorded build, if the request, its inputs relative to `base` and the compiler are
    /// unchanged.
    pub(crate) fn reuse(self, base: &Path, request: &Request) -> Option<(BuiltKernels, Recorded)> {
        let current = self.inputs.request == *request
            && self.requested == requested()
            && self.inputs.current(base)
            && self
                .stamps
                .iter()
                .all(|(path, stamp)| Stamp::of(path).as_ref() == Some(stamp));
        if !current {
            return None;
        }
        let built = self.artifact(base)?;
        Some((built, self.inputs))
    }

    /// The recorded artifact, while its files are intact.
    pub(crate) fn artifact(&self, base: &Path) -> Option<BuiltKernels> {
        if !cache::valid(&self.artifact, self.artifact.file_name()?.to_str()?) {
            return None;
        }
        let abi = std::fs::read_to_string(self.artifact.join(ArtifactFile::Abi.name())).ok()?;
        Some(BuiltKernels {
            directory: self.artifact.clone(),
            abi: Abi::from_json(&abi).ok()?,
            tracked: std::iter::once(self.compiler.clone())
                .chain(self.inputs.paths(base))
                .collect(),
            saved: false,
        })
    }
}
