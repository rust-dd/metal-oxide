use crate::process::Result;
use metal_oxide_artifact::{Abi, ArtifactFile, Manifest, sha256};
use std::path::Path;

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
mod tests {
    use super::*;
    use metal_oxide_artifact::{ABI_VERSION, BuildInfo, DEVICE_TARGET, Files, Kernel, MSL_VERSION};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "oxide-cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn cached_artifact_rejects_tampering_missing_files_and_wrong_identity() {
        let directory = Directory::new();
        let fingerprint = sha256(b"build");
        let hash = sha256(b"file");
        let abi = Abi {
            version: ABI_VERSION,
            required_features: vec![],
            kernels: vec![Kernel {
                name: "empty".into(),
                parameters: vec![],
                required_block: None,
            }],
        };
        let manifest = Manifest {
            abi: abi.clone(),
            target: DEVICE_TARGET.into(),
            msl_version: MSL_VERSION.into(),
            required_features: vec![],
            files: Files {
                msl: hash.clone(),
                oxide_ir: hash.clone(),
                ir: hash.clone(),
                metallib: hash.clone(),
                bindings: hash,
            },
            build: BuildInfo {
                fingerprint: fingerprint.clone(),
                compiler: "oxide".into(),
                rustc: "nightly".into(),
                metal: "metal".into(),
                sdk: "macosx".into(),
                rust_flags: vec![],
                metal_flags: vec![],
            },
        };
        std::fs::write(
            directory.0.join("manifest.json"),
            manifest.to_json().unwrap(),
        )
        .unwrap();
        std::fs::write(directory.0.join("abi.json"), abi.to_json().unwrap()).unwrap();
        let files = [
            "kernels.metal",
            "kernels.oxide-ir",
            "kernels.ir",
            "kernels.metallib",
            "bindings.rs",
        ];
        for name in files {
            std::fs::write(directory.0.join(name), b"file").unwrap();
        }
        assert!(valid(&directory.0, &fingerprint));
        assert!(!valid(&directory.0, &sha256(b"other build")));
        for name in files {
            let path = directory.0.join(name);
            std::fs::write(&path, b"changed").unwrap();
            assert!(
                !valid(&directory.0, &fingerprint),
                "accepted changed {name}"
            );
            std::fs::remove_file(&path).unwrap();
            assert!(
                !valid(&directory.0, &fingerprint),
                "accepted missing {name}"
            );
            std::fs::write(&path, b"file").unwrap();
        }
        let mut abi = abi;
        abi.kernels[0].name = "different".into();
        std::fs::write(directory.0.join("abi.json"), abi.to_json().unwrap()).unwrap();
        assert!(!valid(&directory.0, &fingerprint));
    }
}
