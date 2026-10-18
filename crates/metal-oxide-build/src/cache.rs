use crate::process::Result;
use metal_oxide_artifact::{ArtifactFile, Manifest, sha256};
use std::{path::Path, time::SystemTime};

/// Dates the files that hosts embed back to 1970: Cargo takes a file written during a build
/// for a change and would build the host once more, and an artifact never changes in place.
pub(crate) fn settle(directory: &Path) -> Result<()> {
    for file in [ArtifactFile::Manifest, ArtifactFile::Metallib] {
        let file = std::fs::File::options()
            .write(true)
            .open(directory.join(file.name()))?;
        file.set_modified(SystemTime::UNIX_EPOCH)?;
    }
    Ok(())
}

pub(crate) fn discard_abandoned(root: &Path) -> Result<()> {
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some((pid, sequence)) = name
            .to_str()
            .and_then(|name| name.strip_prefix(".build-"))
            .and_then(|name| name.split_once('-'))
        else {
            continue;
        };
        if [pid, sequence]
            .iter()
            .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
            && entry.file_type()?.is_dir()
        {
            std::fs::remove_dir_all(entry.path())?;
        }
    }
    Ok(())
}

/// The manifest of the artifact cached in `directory`, while it records `fingerprint` and every
/// file it hashes is intact.
pub(crate) fn manifest(directory: &Path, fingerprint: &str) -> Option<Manifest> {
    let text = std::fs::read_to_string(directory.join(ArtifactFile::Manifest.name())).ok()?;
    let manifest = Manifest::from_json(&text).ok()?;
    let intact = manifest.build.fingerprint == fingerprint
        && manifest
            .files
            .entries()
            .into_iter()
            .all(|(file, expected)| {
                std::fs::read(directory.join(file.name()))
                    .is_ok_and(|bytes| sha256(&bytes) == expected)
            });
    intact.then_some(manifest)
}

#[cfg(test)]
mod tests;
