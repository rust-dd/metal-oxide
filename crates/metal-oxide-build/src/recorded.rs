use crate::{
    configuration::Configuration,
    environment::Environment,
    metadata::{Metadata, Project},
    process::Result,
};
use metal_oxide_artifact::{MathMode, sha256};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

#[cfg(test)]
mod tests;

/// Increments when recording changes, which makes older records stale.
pub(crate) const FORMAT: u32 = 4;

/// What a kernel build is asked for besides its packages' files.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    /// SHA-256 of what names the kernels with its file and item: the source that kernels
    /// written in a host crate copy, or a kernel crate's path.
    pub(crate) source: String,
    pub(crate) math_mode: MathMode,
    /// Line tables and sources for a GPU capture.
    pub(crate) debug_info: bool,
}

/// The inputs an artifact was built from. Paths are relative to the host package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Recorded {
    pub(crate) format: u32,
    /// The fingerprint in the artifact's manifest.
    pub(crate) artifact: String,
    pub(crate) request: Request,
    pub(crate) profile: BTreeMap<String, String>,
    pub(crate) configuration: Configuration,
    pub(crate) environment: Environment,
    pub(crate) lock: String,
    /// The kernels' dependencies from the lock file as `name version source`, or as
    /// `name version` for path packages.
    pub(crate) packages: Vec<String>,
    /// SHA-256 of each manifest and source file of the path packages that the build read.
    pub(crate) files: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(tag = "reason")]
enum Message {
    #[serde(rename = "compiler-artifact")]
    Artifact {
        manifest_path: PathBuf,
        filenames: Vec<PathBuf>,
    },
    #[serde(other)]
    Other,
}

impl Recorded {
    /// Records the packages and files in Cargo's messages from building `project`'s kernels,
    /// skipping the toolchain under `sysroot`. For a crate copied from a host crate, `host`
    /// resolves the packages it names by path and its lock file pins them; the copied crate's
    /// own files are left out.
    pub(crate) fn from_build(
        project: &Project,
        host: Option<&Metadata>,
        messages: &str,
        sysroot: &Path,
        base: &Path,
        artifact: &str,
        request: Request,
    ) -> Result<Self> {
        let copied = host.is_some();
        let host = host.unwrap_or(&project.metadata);
        let lock = host.workspace_root.join("Cargo.lock");
        let entries = lock_entries(&std::fs::read_to_string(&lock)?);
        let kernel = &project.kernel().manifest_path;
        let mut packages = Vec::new();
        let mut files = BTreeMap::new();
        let configuration = Configuration::record(&project.metadata.workspace_root, base, copied)?;
        let mut environment = Environment::default();
        if !copied {
            let workspace = project.metadata.workspace_root.join("Cargo.toml");
            files.insert(
                relative(&workspace, base).to_string_lossy().into_owned(),
                sha256(&std::fs::read(workspace)?),
            );
        }
        for line in messages.lines() {
            let Message::Artifact {
                manifest_path,
                filenames,
            } = serde_json::from_str(line)?
            else {
                continue;
            };
            let mut sources = Vec::new();
            for output in filenames
                .iter()
                .filter(|file| file.extension() == Some("rmeta".as_ref()))
            {
                let stem = output.file_stem().unwrap().to_string_lossy();
                let info = output
                    .with_file_name(format!("{}.d", stem.strip_prefix("lib").unwrap_or(&stem)));
                let info = std::fs::read_to_string(&info)?;
                environment.record(&info, &configuration.environment)?;
                sources.extend(dependencies(&info, &project.metadata.workspace_root));
            }
            if manifest_path.starts_with(sysroot) || copied && manifest_path == *kernel {
                continue;
            }
            let package = host
                .packages
                .iter()
                .find(|package| package.manifest_path == manifest_path)
                .ok_or_else(|| {
                    format!(
                        "{} is missing from the host's Cargo metadata",
                        manifest_path.display()
                    )
                })?;
            let local = package.id.starts_with("path+");
            // A path dependency that becomes a git or registry package changes its lock entry.
            if manifest_path != *kernel {
                let prefix = if local {
                    format!("{} {}", package.name, package.version)
                } else {
                    let source = package.id.rsplit_once('#').map_or("", |(source, _)| source);
                    format!("{} {} {source}", package.name, package.version)
                };
                let found = entries
                    .iter()
                    .filter(|entry| {
                        if local {
                            **entry == prefix
                        } else {
                            entry.starts_with(&prefix)
                        }
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                if found.is_empty() {
                    return Err(format!("{} is missing from {}", package.id, lock.display()).into());
                }
                packages.extend(found);
            }
            if !local {
                continue;
            }
            sources.push(manifest_path);
            for source in sources
                .into_iter()
                .filter(|path| !path.starts_with(sysroot))
            {
                let key = relative(&source, base).to_string_lossy().into_owned();
                files.insert(key, sha256(&std::fs::read(&source)?));
            }
        }
        packages.sort();
        packages.dedup();
        Ok(Self {
            format: FORMAT,
            artifact: artifact.into(),
            request,
            profile: profile(),
            configuration,
            environment,
            lock: relative(&lock, base).to_string_lossy().into_owned(),
            packages,
            files,
        })
    }

    /// Whether every recorded file and package is unchanged, with `base` as the kernel package.
    pub(crate) fn current(&self, base: &Path) -> bool {
        if self.format != FORMAT
            || self.profile != profile()
            || !self.configuration.current(base)
            || !self.environment.current()
        {
            return false;
        }
        let files = self.files.iter().all(|(path, hash)| {
            std::fs::read(base.join(path)).is_ok_and(|bytes| sha256(&bytes) == *hash)
        });
        files
            && std::fs::read_to_string(base.join(&self.lock)).is_ok_and(|lock| {
                let entries = lock_entries(&lock);
                self.packages
                    .iter()
                    .all(|package| entries.contains(package))
            })
    }

    /// The recorded files and lock file, for Cargo's change tracking.
    pub(crate) fn paths(&self, base: &Path) -> Vec<PathBuf> {
        self.files
            .keys()
            .chain([&self.lock])
            .map(|path| normalize(&base.join(path)))
            .chain(self.configuration.paths(base))
            .collect()
    }
}

pub(crate) fn profile() -> BTreeMap<String, String> {
    std::env::vars()
        .filter(|(key, _)| key.starts_with("CARGO_PROFILE_RELEASE_"))
        .collect()
}

/// Each `[[package]]` of a Cargo.lock as `name version source`, or `name version` without a
/// source.
fn lock_entries(lock: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut fields = [None, None, None];
    for line in lock.lines().map(str::trim).chain(["[[package]]"]) {
        if line.starts_with('[') && !line.starts_with("[[package]]") || line == "[[package]]" {
            if let [Some(name), Some(version), source] = std::mem::take(&mut fields) {
                entries.push(match source {
                    Some(source) => format!("{name} {version} {source}"),
                    None => format!("{name} {version}"),
                });
            }
            continue;
        }
        for (index, key) in ["name", "version", "source"].into_iter().enumerate() {
            if let Some(value) = line
                .strip_prefix(key)
                .and_then(|v| v.trim_start().strip_prefix('='))
            {
                fields[index] = Some(value.trim().trim_matches('"').to_owned());
            }
        }
    }
    entries
}

/// The dependencies of the first rule in a rustc dep-info file, resolved against `root`.
pub(crate) fn dependencies(info: &str, root: &Path) -> Vec<PathBuf> {
    let Some((_, rest)) = info.lines().next().and_then(|line| line.split_once(": ")) else {
        return vec![];
    };
    let mut paths = Vec::new();
    let mut current = String::new();
    let mut characters = rest.chars();
    while let Some(character) = characters.next() {
        match character {
            '\\' => current.extend(characters.next()),
            ' ' => paths.push(std::mem::take(&mut current)),
            _ => current.push(character),
        }
    }
    paths.push(current);
    paths
        .into_iter()
        .filter(|path| !path.is_empty())
        .map(|path| normalize(&root.join(path)))
        .collect()
}

pub(crate) fn normalize(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir
                if !matches!(
                    result.components().next_back(),
                    None | Some(Component::ParentDir)
                ) =>
            {
                result.pop();
            }
            Component::CurDir => {}
            component => result.push(component),
        }
    }
    result
}

/// `path` relative to the directory `base`; both are absolute.
pub(crate) fn relative(path: &Path, base: &Path) -> PathBuf {
    let (path, base) = (normalize(path), normalize(base));
    let common = path
        .components()
        .zip(base.components())
        .take_while(|(a, b)| a == b)
        .count();
    base.components()
        .skip(common)
        .map(|_| Component::ParentDir.as_os_str())
        .chain(path.components().skip(common).map(|c| c.as_os_str()))
        .collect()
}
