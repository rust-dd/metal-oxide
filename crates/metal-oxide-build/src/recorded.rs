use crate::{metadata::Project, process::Result};
use metal_oxide_artifact::{MathMode, sha256};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Component, Path, PathBuf},
};

#[cfg(test)]
mod tests;

/// The inputs a committed artifact was built from. Paths are relative to the kernel package.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Recorded {
    /// The fingerprint in the artifact's manifest.
    pub(crate) artifact: String,
    pub(crate) math_mode: MathMode,
    pub(crate) lock: String,
    /// Registry and git packages as `name version source` from the lock file.
    pub(crate) packages: Vec<String>,
    /// SHA-256 of each manifest and source file that the kernel build read.
    pub(crate) files: BTreeMap<String, String>,
}

#[derive(Deserialize)]
#[serde(tag = "reason")]
enum Message {
    #[serde(rename = "compiler-artifact")]
    Artifact {
        package_id: String,
        manifest_path: PathBuf,
        filenames: Vec<PathBuf>,
    },
    #[serde(other)]
    Other,
}

impl Recorded {
    /// Records the packages and files in Cargo's messages, skipping the toolchain under `sysroot`.
    pub(crate) fn from_build(
        project: &Project,
        messages: &str,
        sysroot: &Path,
        artifact: &str,
        math_mode: MathMode,
    ) -> Result<Self> {
        let base = project.kernel().manifest_path.parent().unwrap();
        let root = &project.metadata.workspace_root;
        let lock = root.join("Cargo.lock");
        let entries = lock_entries(&std::fs::read_to_string(&lock)?);
        let mut packages = Vec::new();
        let mut files = BTreeMap::new();
        for line in messages.lines() {
            let Message::Artifact {
                package_id,
                manifest_path,
                filenames,
            } = serde_json::from_str(line)?
            else {
                continue;
            };
            if manifest_path.starts_with(sysroot) {
                continue;
            }
            if !package_id.starts_with("path+") {
                let (source, package) = package_id
                    .rsplit_once('#')
                    .ok_or_else(|| format!("unexpected Cargo package id {package_id}"))?;
                let (name, version) = package
                    .split_once('@')
                    .ok_or_else(|| format!("unexpected Cargo package id {package_id}"))?;
                let found = entries
                    .iter()
                    .filter(|entry| entry.starts_with(&format!("{name} {version} {source}")))
                    .cloned()
                    .collect::<Vec<_>>();
                if found.is_empty() {
                    return Err(format!("{package_id} is missing from {}", lock.display()).into());
                }
                packages.extend(found);
                continue;
            }
            let mut sources = vec![manifest_path];
            for output in filenames
                .iter()
                .filter(|f| f.extension() == Some("rmeta".as_ref()))
            {
                let stem = output.file_stem().unwrap().to_string_lossy();
                let info = output
                    .with_file_name(format!("{}.d", stem.strip_prefix("lib").unwrap_or(&stem)));
                sources.extend(dependencies(&std::fs::read_to_string(&info)?, root));
            }
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
            artifact: artifact.into(),
            math_mode,
            lock: relative(&lock, base).to_string_lossy().into_owned(),
            packages,
            files,
        })
    }

    /// Whether every recorded file and package is unchanged, with `base` as the kernel package.
    pub(crate) fn current(&self, base: &Path) -> bool {
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
            .collect()
    }
}

/// Each `[[package]]` of a Cargo.lock as `name version source`.
fn lock_entries(lock: &str) -> Vec<String> {
    let mut entries = Vec::new();
    let mut fields = [None, None, None];
    for line in lock.lines().map(str::trim).chain(["[[package]]"]) {
        if line.starts_with('[') && !line.starts_with("[[package]]") || line == "[[package]]" {
            if let [Some(name), Some(version), source] = std::mem::take(&mut fields) {
                entries.push(format!("{name} {version} {}", source.unwrap_or_default()));
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
fn dependencies(info: &str, root: &Path) -> Vec<PathBuf> {
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

fn normalize(path: &Path) -> PathBuf {
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
fn relative(path: &Path, base: &Path) -> PathBuf {
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
