use crate::{metadata::Project, process::Result};
use metal_oxide_artifact::{Abi, Manifest, sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

fn files(directory: &Path, output: &mut BTreeMap<PathBuf, String>) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        if ["target", ".git", ".zed", "docs"]
            .iter()
            .any(|name| entry.file_name() == *name)
        {
            continue;
        }
        let kind = entry.file_type()?;
        if kind.is_dir() {
            files(&entry.path(), output)?;
        } else if kind.is_file() {
            output.insert(entry.path(), sha256(&std::fs::read(entry.path())?));
        } else if kind.is_symlink() {
            return Err(format!(
                "symlinked kernel source is unsupported: {}",
                entry.path().display()
            )
            .into());
        }
    }
    Ok(())
}

pub(crate) fn sources(project: &Project) -> Result<BTreeMap<PathBuf, String>> {
    let mut result = BTreeMap::new();
    for package in &project.metadata.packages {
        if project.dependencies.contains(&package.id) {
            files(package.manifest_path.parent().unwrap(), &mut result)?;
        }
    }
    for name in ["Cargo.toml", "Cargo.lock"] {
        let path = project.metadata.workspace_root.join(name);
        result.insert(path.clone(), sha256(&std::fs::read(path)?));
    }
    for ancestor in project.metadata.workspace_root.ancestors() {
        for name in [".cargo/config", ".cargo/config.toml"] {
            let path = ancestor.join(name);
            if path.is_file() {
                result.insert(path.clone(), sha256(&std::fs::read(path)?));
            }
        }
    }
    Ok(result)
}

pub(crate) fn valid(directory: &Path, fingerprint: &str) -> bool {
    let validate = || -> Result<()> {
        let manifest =
            Manifest::from_json(&std::fs::read_to_string(directory.join("manifest.json"))?)?;
        if manifest.build.fingerprint != fingerprint {
            return Err("artifact fingerprint mismatch".into());
        }
        let abi = Abi::from_json(&std::fs::read_to_string(directory.join("abi.json"))?)?;
        if abi != manifest.abi {
            return Err("cached ABI differs from manifest".into());
        }
        for (name, expected) in [
            ("kernels.metal", &manifest.files.msl),
            ("kernels.oxide-ir", &manifest.files.oxide_ir),
            ("kernels.ir", &manifest.files.ir),
            ("kernels.metallib", &manifest.files.metallib),
            ("bindings.rs", &manifest.files.bindings),
        ] {
            if sha256(&std::fs::read(directory.join(name))?) != *expected {
                return Err(format!("artifact hash mismatch: {name}").into());
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

    #[test]
    fn source_hashes_track_nested_changes() {
        let directory = Directory::new();
        std::fs::create_dir(directory.0.join("src")).unwrap();
        let source = directory.0.join("src/lib.rs");
        std::fs::write(&source, "pub fn kernel() {}").unwrap();
        let mut before = BTreeMap::new();
        files(&directory.0, &mut before).unwrap();
        std::fs::write(&source, "pub fn kernel() { let x = 1; }").unwrap();
        let mut after = BTreeMap::new();
        files(&directory.0, &mut after).unwrap();
        assert_ne!(before[&source], after[&source]);
    }
}
