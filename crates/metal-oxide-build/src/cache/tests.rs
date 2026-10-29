use super::*;
use metal_oxide_artifact::{ABI_VERSION, Abi, BuildInfo, Files, Kernel};
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
        math_mode: metal_oxide_artifact::MathMode::Precise,
        required_features: vec![],
        kernels: vec![Kernel {
            name: "empty".into(),
            parameters: vec![],
            required_block: None,
            gate_binding: 0,
            fallible: false,
            lengths: false,
            one_dimensional: false,
            safe: false,
            requires: Vec::new(),
        }],
    };
    let metal_flags = abi
        .target_flags()
        .into_iter()
        .chain(abi.math_mode.metal_flags().map(str::to_owned))
        .collect::<Vec<_>>();
    let written = Manifest {
        abi,
        checks: vec![],
        files: Files {
            msl: hash.clone(),
            oxide_ir: hash.clone(),
            rustc_args: hash.clone(),
            ir: hash.clone(),
            metallib: hash,
        },
        build: BuildInfo {
            fingerprint: fingerprint.clone(),
            compiler: "oxide".into(),
            rustc: "nightly".into(),
            metal: "metal".into(),
            sdk: "macosx".into(),
            metal_flags,
        },
    };
    std::fs::write(
        directory.0.join("manifest.json"),
        written.to_json().unwrap(),
    )
    .unwrap();
    let files = [
        "kernels.metal",
        "kernels.oxide-ir",
        "rustc-args.json",
        "kernels.ir",
        "kernels.metallib",
    ];
    for name in files {
        std::fs::write(directory.0.join(name), b"file").unwrap();
    }
    assert_eq!(manifest(&directory.0, &fingerprint), Some(written));
    assert!(manifest(&directory.0, &sha256(b"other build")).is_none());
    for name in files {
        let path = directory.0.join(name);
        std::fs::write(&path, b"changed").unwrap();
        assert!(
            manifest(&directory.0, &fingerprint).is_none(),
            "accepted changed {name}"
        );
        std::fs::remove_file(&path).unwrap();
        assert!(
            manifest(&directory.0, &fingerprint).is_none(),
            "accepted missing {name}"
        );
        std::fs::write(&path, b"file").unwrap();
    }
    std::fs::write(directory.0.join("manifest.json"), b"{}").unwrap();
    assert!(manifest(&directory.0, &fingerprint).is_none());
}
