use crate::process::Result;
use metal_oxide_artifact::{Abi, ArtifactFile, Manifest, sha256};
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

pub(crate) fn valid(directory: &Path, fingerprint: &str) -> bool {
    let validate = || -> Result<()> {
        let manifest = Manifest::from_json(&std::fs::read_to_string(
            directory.join(ArtifactFile::Manifest.name()),
        )?)?;
        if manifest.build.fingerprint != fingerprint {
            return Err("artifact fingerprint mismatch".into());
        }
        let abi = Abi::from_json(&std::fs::read_to_string(
            directory.join(ArtifactFile::Abi.name()),
        )?)?;
        if abi != manifest.abi {
            return Err("cached ABI differs from manifest".into());
        }
        let arguments: Vec<String> = serde_json::from_slice(&std::fs::read(
            directory.join(ArtifactFile::RustcArgs.name()),
        )?)?;
        if arguments != manifest.build.rustc_args {
            return Err("cached compiler arguments differ from manifest".into());
        }
        for (file, expected) in manifest.files.entries() {
            if sha256(&std::fs::read(directory.join(file.name()))?) != expected {
                return Err(format!("artifact hash mismatch: {}", file.name()).into());
            }
        }
        Ok(())
    };
    validate().is_ok()
}

#[cfg(test)]
mod tests;
