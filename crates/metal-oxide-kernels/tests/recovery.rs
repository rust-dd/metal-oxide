#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide_artifact::{Abi, ArtifactFile, COMPILER_OUTPUTS, Manifest, sha256};
use std::{
    os::unix::{fs::PermissionsExt, process::CommandExt},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};
use support::{Project, checked};

const KERNEL: &str = r#"
use metal_oxide::device::{DisjointBuffer, thread_index};

#[metal_oxide::kernel]
pub fn recover(mut out: DisjointBuffer<u32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        *out = 7;
    }
}

fn main() {}
"#;

/// Holds or fails the Metal compiler in `TEST_PHASE`, compile (`-c`) or link; passes every
/// other `xcrun` call through.
const XCRUN: &str = r#"#!/bin/sh
for argument; do [ "$argument" = metal ] && metal=1; done
[ -n "$metal" ] || exec /usr/bin/xcrun "$@"
case " $* " in *" -c "*) phase=compile ;; *) phase=link ;; esac
case " $* " in *" --version "*) phase=none ;; esac
if [ "$phase" = "$TEST_PHASE" ]; then
    if [ "$TEST_ACTION" = fail ]; then
        echo "intentional Metal compiler failure" >&2
        exit 7
    fi
    echo ready > "$TEST_MARKER"
    exec sleep 600
fi
exec /usr/bin/xcrun "$@"
"#;

fn entries(root: &Path, prefix: &str) -> Vec<PathBuf> {
    std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(prefix)
        })
        .collect()
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn interrupted_metal_builds_recover_and_failed_ones_keep_their_inputs() {
    let project = Project::new(KERNEL);
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let artifact = project.artifact("recover");
    let root = artifact.parent().unwrap();
    let shims = project.0.join("shims");
    std::fs::create_dir(&shims).unwrap();
    std::fs::write(shims.join("xcrun"), XCRUN).unwrap();
    std::fs::set_permissions(shims.join("xcrun"), std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = std::env::var_os("PATH").unwrap();
    let path = std::env::join_paths(std::iter::once(shims).chain(std::env::split_paths(&path)));
    let shimmed = |phase: &str, action: &str| {
        let mut command = project.cargo(&["build", "--quiet"]);
        command
            .env("PATH", path.as_ref().unwrap())
            .env("TEST_PHASE", phase)
            .env("TEST_ACTION", action)
            .env("TEST_MARKER", project.0.join("held"));
        command
    };

    for phase in ["compile", "link"] {
        // A damaged artifact makes the next build compile it again.
        std::fs::remove_file(artifact.join("kernels.metallib")).unwrap();
        let _ = std::fs::remove_file(project.0.join("held"));
        let mut child = shimmed(phase, "hold").process_group(0).spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(300);
        while !project.0.join("held").exists() {
            assert!(
                child.try_wait().unwrap().is_none(),
                "the build ended before {phase}"
            );
            assert!(Instant::now() < deadline, "the build never reached {phase}");
            std::thread::sleep(Duration::from_millis(50));
        }
        let group = format!("-{}", child.id());
        checked(
            Command::new("kill")
                .args(["-KILL", &group])
                .output()
                .unwrap(),
        );
        child.wait().unwrap();
        assert!(
            !entries(root, ".build-").is_empty(),
            "{phase}: no stage left behind"
        );

        checked(project.cargo(&["build", "--quiet"]).output().unwrap());
        assert_eq!(project.artifact("recover"), artifact);
        assert!(artifact.join("kernels.metallib").is_file());
        assert!(
            entries(root, ".build-").is_empty(),
            "{phase}: an abandoned stage survived"
        );
    }

    std::fs::remove_file(artifact.join("kernels.metallib")).unwrap();
    let before = entries(root, ".failed-");
    let output = shimmed("compile", "fail").output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("inputs retained at"));
    let failed = entries(root, ".failed-")
        .into_iter()
        .filter(|stage| !before.contains(stage))
        .collect::<Vec<_>>();
    let [stage] = failed.as_slice() else {
        panic!("expected one failed stage, found {failed:?}");
    };
    for file in [
        "kernels.metal",
        "kernels.oxide-ir",
        "abi.json",
        "metal-commands.json",
    ] {
        assert!(stage.join(file).is_file(), "{file}");
    }
    let log = std::fs::read_to_string(stage.join("metal.log")).unwrap();
    assert!(log.contains("intentional Metal compiler failure"));
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    assert_eq!(project.artifact("recover"), artifact);
    assert!(stage.exists(), "a retry discarded the failure's inputs");
    std::fs::remove_dir_all(stage).unwrap();
}

/// Edits `src/main.rs` outside the kernel, so the next build expands the macro again.
fn edit(project: &Project, edit: usize) {
    project.write(&format!("{KERNEL}\n// edit {edit}\n"));
}

fn validate(directory: &Path) {
    let manifest = std::fs::read_to_string(directory.join("manifest.json")).unwrap();
    let manifest = Manifest::from_json(&manifest).unwrap();
    let abi = std::fs::read_to_string(directory.join("abi.json")).unwrap();
    assert_eq!(manifest.abi, Abi::from_json(&abi).unwrap());
    for (file, hash) in manifest.files.entries() {
        assert_eq!(
            sha256(&std::fs::read(directory.join(file.name())).unwrap()),
            hash
        );
    }
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn damaged_artifacts_are_rebuilt_and_abandoned_stages_removed() {
    let project = Project::new(KERNEL);
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let directory = project.artifact("recover");
    let outputs =
        COMPILER_OUTPUTS.map(|file| (file, std::fs::read(directory.join(file.name())).unwrap()));
    let root = directory.parent().unwrap();
    let abandoned = root.join(".build-999999-0");
    std::fs::create_dir(&abandoned).unwrap();
    std::fs::write(abandoned.join("kernels.metallib"), b"partial").unwrap();
    let unrelated = root.join(format!(".build-not-owned-{}", std::process::id()));
    std::fs::create_dir(&unrelated).unwrap();
    let mut edits = 0;
    for file in [
        ArtifactFile::Msl,
        ArtifactFile::OxideIr,
        ArtifactFile::Air,
        ArtifactFile::Metallib,
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
            edits += 1;
            edit(&project, edits);
            checked(project.cargo(&["build", "--quiet"]).output().unwrap());
            assert_eq!(project.artifact("recover"), directory);
            validate(&directory);
            for (file, expected) in &outputs {
                assert_eq!(
                    &std::fs::read(directory.join(file.name())).unwrap(),
                    expected
                );
            }
        }
    }
    assert!(!abandoned.exists(), "an interrupted build's stage survived");
    assert!(unrelated.exists());
    std::fs::remove_dir(unrelated).unwrap();
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn tampered_compiler_outputs_are_regenerated() {
    let project = Project::new(KERNEL);
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let artifact = project.artifact("recover");
    let expected = std::fs::read(artifact.join("kernels.metal")).unwrap();
    let mut tampered = 0;
    for configuration in std::fs::read_dir(Project::target().join("metal/rust")).unwrap() {
        for output in std::fs::read_dir(configuration.unwrap().path()).unwrap() {
            let msl = output.unwrap().path().join("kernels.metal");
            if std::fs::read(&msl).is_ok_and(|msl| msl == expected) {
                let corrupted =
                    "#include <metal_stdlib>\nusing namespace metal;\nkernel void corrupted() {}\n";
                std::fs::write(&msl, corrupted).unwrap();
                tampered += 1;
            }
        }
    }
    assert!(tampered > 0);
    // A damaged artifact makes the next build compile the kernels again.
    std::fs::remove_file(artifact.join("kernels.metallib")).unwrap();
    edit(&project, 1);
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let rebuilt = project.artifact("recover");
    assert_eq!(
        std::fs::read(rebuilt.join("kernels.metal")).unwrap(),
        expected
    );
}
