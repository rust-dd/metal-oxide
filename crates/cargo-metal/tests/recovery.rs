mod support;

use metal_oxide_artifact::{Abi, ArtifactFile, COMPILER_OUTPUTS, Manifest, sha256};
use std::path::PathBuf;
use support::{Workspace, checked};

fn build(workspace: &Workspace) -> (PathBuf, String) {
    let output = checked(workspace.command("build").output().unwrap());
    let log = String::from_utf8(output.stderr).unwrap();
    let path = log
        .lines()
        .find_map(|line| {
            line.strip_prefix("Built Metal artifact ")
                .or_else(|| line.strip_prefix("Cached Metal artifact "))
        })
        .unwrap();
    (PathBuf::from(path), log)
}

fn validate(directory: &std::path::Path) {
    let manifest =
        Manifest::from_json(&std::fs::read_to_string(directory.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(
        manifest.abi,
        Abi::from_json(&std::fs::read_to_string(directory.join("abi.json")).unwrap()).unwrap()
    );
    for (file, hash) in manifest.files.entries() {
        assert_eq!(
            sha256(&std::fs::read(directory.join(file.name())).unwrap()),
            hash
        );
    }
}

#[test]
#[ignore = "requires pinned compiler and Metal toolchain"]
fn cache_recovers_each_missing_or_corrupt_file_and_abandoned_stage() {
    let workspace = Workspace::new();
    let (directory, _) = build(&workspace);
    let outputs =
        COMPILER_OUTPUTS.map(|file| (file, std::fs::read(directory.join(file.name())).unwrap()));
    let root = directory.parent().unwrap();
    let abandoned = root.join(".build-999999-0");
    std::fs::create_dir(&abandoned).unwrap();
    std::fs::write(abandoned.join("kernels.metallib"), b"partial").unwrap();
    let unrelated = root.join(".build-not-owned");
    std::fs::create_dir(&unrelated).unwrap();
    let (_, log) = build(&workspace);
    assert!(log.contains("Cached Metal artifact"));
    assert!(
        !abandoned.exists(),
        "interrupted build stage was not removed"
    );
    assert!(unrelated.exists());
    for file in [
        ArtifactFile::Msl,
        ArtifactFile::OxideIr,
        ArtifactFile::Air,
        ArtifactFile::Metallib,
        ArtifactFile::Bindings,
        ArtifactFile::RustcArgs,
        ArtifactFile::Abi,
        ArtifactFile::Manifest,
    ] {
        for missing in [false, true] {
            let path = directory.join(file.name());
            if missing {
                std::fs::remove_file(path).unwrap();
            } else {
                std::fs::write(path, b"corrupt").unwrap();
            }
            let (rebuilt, log) = build(&workspace);
            assert_eq!(rebuilt, directory);
            assert!(log.contains("Built Metal artifact"), "{log}");
            validate(&directory);
            for (file, expected) in &outputs {
                assert_eq!(
                    &std::fs::read(directory.join(file.name())).unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
#[ignore = "requires pinned compiler and Metal toolchain"]
fn cached_rust_outputs_are_regenerated_after_tampering() {
    let workspace = Workspace::new();
    let (artifact, _) = build(&workspace);
    let expected = std::fs::read(artifact.join("kernels.metal")).unwrap();
    let root = workspace.0.join("target/metal/rust");
    let rust = std::fs::read_dir(root)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let output = std::fs::read_dir(rust)
        .unwrap()
        .map(|e| e.unwrap())
        .find(|e| e.file_name().to_string_lossy().starts_with("output-"))
        .unwrap()
        .path();
    std::fs::write(
        output.join("kernels.metal"),
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void corrupted() {}\n",
    )
    .unwrap();
    let (rebuilt, _) = build(&workspace);
    assert_eq!(
        std::fs::read(rebuilt.join("kernels.metal")).unwrap(),
        expected
    );
}
