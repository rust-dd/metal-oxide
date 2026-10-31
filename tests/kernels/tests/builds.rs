#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use metal_oxide_artifact::MathMode;
use std::{os::unix::fs::PermissionsExt, path::Path, process::Command};
use support::{Project, checked, failure};

const KERNEL: &str = r#"
use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

#[metal_oxide::kernel]
pub fn scale(a: ReadBuffer<u32>, mut out: DisjointBuffer<u32>) {
    let index = thread_index();
    let a = a.as_slice();
    if let Some(out) = out.get_mut(&index)
        && index.get() < a.len()
    {
        *out = a[index.get()] * FACTOR;
    }
}

fn main() {}
"#;

fn source(factor: u32) -> String {
    KERNEL.replace("FACTOR", &factor.to_string())
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn unchanged_kernels_reuse_their_artifact_and_edits_or_damage_rebuild() {
    let project = Project::new(&source(2));
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let first = project.artifact("scale");

    project.write(&format!("{}\n// an edit outside the kernel\n", source(2)));
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    assert_eq!(project.artifact("scale"), first);

    project.write(&source(3));
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let second = project.artifact("scale");
    assert_ne!(second, first);
    // The artifact that a build writes leaves the next build fresh.
    let output = checked(
        project
            .cargo(&["build", "--message-format=json"])
            .output()
            .unwrap(),
    );
    let messages = String::from_utf8_lossy(&output.stdout);
    let host = messages
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|message| {
            message["reason"] == "compiler-artifact" && message["target"]["name"] == "host"
        })
        .unwrap();
    assert_eq!(host["fresh"], true);
    assert!(
        std::fs::read_to_string(second.join("kernels.oxide-ir"))
            .unwrap()
            .contains('3')
    );

    let library = second.join("kernels.metallib");
    let bytes = std::fs::read(&library).unwrap();
    std::fs::write(&library, b"damaged").unwrap();
    project.write(&format!("{}\n// another edit\n", source(3)));
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    assert_eq!(
        std::fs::read(project.artifact("scale").join("kernels.metallib")).unwrap(),
        bytes
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn math_modes_build_separate_artifacts() {
    let module = |mode: &str| {
        format!(
            "mod {mode} {{\n    use metal_oxide::device::*;\n    #[kernel]\n    pub fn root(mut out: DisjointBuffer<f32>) {{\n        let index = thread_index();\n        if let Some(out) = out.get_mut(&index) {{\n            *out = (index.get() as f32).sqrt();\n        }}\n    }}\n}}\n"
        )
    };
    let main = format!(
        "#[metal_oxide::kernels]\n{}#[metal_oxide::kernels(math = \"fast\")]\n{}fn main() {{}}\n",
        module("precise"),
        module("fast")
    );
    let project = Project::new(&main);
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let precise = project.artifact("precise");
    let fast = project.artifact_of("fast", MathMode::Fast, false);
    let mode = |artifact: &Path| {
        let manifest = std::fs::read_to_string(artifact.join("manifest.json")).unwrap();
        metal_oxide_artifact::Manifest::from_json(&manifest)
            .unwrap()
            .abi
            .math_mode
    };
    assert_eq!(mode(&precise), MathMode::Precise);
    assert_eq!(mode(&fast), MathMode::Fast);
    for saved in ["precise-precise", "fast-fast"] {
        assert!(
            project
                .0
                .join("metal/src/main.rs")
                .join(saved)
                .join("inputs.json")
                .is_file()
        );
    }
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn release_overflow_settings_invalidate_cached_and_saved_kernels() {
    let main = r#"
use metal_oxide::device::{DisjointBuffer, thread_index};
#[metal_oxide::kernel]
pub fn add(mut out: DisjointBuffer<u32>, value: u32) {
    if let Some(out) = out.get_mut(&thread_index()) {
        *out = value + 1;
    }
}
fn main() {}
"#;
    let project = Project::new(main);
    let build = |enabled: bool| {
        checked(
            project
                .cargo(&["build", "--quiet"])
                .env("CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS", enabled.to_string())
                .output()
                .unwrap(),
        );
        let directory = project.artifact("add");
        let text = std::fs::read_to_string(directory.join("manifest.json")).unwrap();
        let manifest = metal_oxide_artifact::Manifest::from_json(&text).unwrap();
        (directory, manifest.checks)
    };
    let (wrapping, checks) = build(false);
    assert!(checks.is_empty(), "{checks:?}");
    project.write(&format!("{main}\n// host edit\n"));
    let (checked, checks) = build(true);
    assert_ne!(
        checked, wrapping,
        "the cached artifact kept wrapping arithmetic"
    );
    assert_eq!(checks.len(), 1, "{checks:?}");
    let saved = project
        .0
        .join("metal/src/main.rs/add-precise/manifest.json");
    let saved = metal_oxide_artifact::Manifest::from_json(&std::fs::read_to_string(saved).unwrap())
        .unwrap();
    assert_eq!(saved.checks, checks);
    let (restored, checks) = build(false);
    assert_eq!(restored, wrapping);
    assert!(checks.is_empty(), "{checks:?}");
}

#[test]
#[ignore = "builds a separate project"]
fn rust_analyzer_expansions_need_no_kernel_build() {
    let project = Project::new(&source(2));
    let target = project.0.join("analysis");
    checked(
        project
            .cargo(&["check", "--quiet"])
            .env("RUST_ANALYZER_INTERNALS_DO_NOT_USE", "this is unstable")
            .env("CARGO_TARGET_DIR", &target)
            .output()
            .unwrap(),
    );
    assert!(!target.join("metal").exists());
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn device_errors_point_at_their_source_line() {
    let main = r#"
use metal_oxide::device::{DisjointBuffer, thread_index};

#[metal_oxide::kernel]
pub fn identify(mut out: DisjointBuffer<u32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        *out = std::process::id();
    }
}

fn main() {}
"#;
    let project = Project::new(main);
    let output = project.cargo(&["build"]).output().unwrap();
    assert!(!output.status.success());
    let errors = String::from_utf8_lossy(&output.stderr);
    assert!(errors.contains("error: GPU build:"), "{errors}");
    assert!(errors.contains("src/main.rs:8:16"), "{errors}");
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn errors_in_core_point_at_the_kernel_call_that_reached_them() {
    let main = r#"
use metal_oxide::device::{DisjointBuffer, ReadBuffer, thread_index};

#[metal_oxide::kernel(requires(input.len() == out.len()))]
pub fn total(input: ReadBuffer<[u32; 2]>, mut out: DisjointBuffer<u32>) {
    let index = thread_index();
    if let Some(out) = out.get_mut(&index) {
        *out = input.as_slice()[index.get()].into_iter().sum::<u32>();
    }
}

fn main() {}
"#;
    let project = Project::new(main);
    let output = project.cargo(&["build"]).output().unwrap();
    assert!(!output.status.success());
    let errors = String::from_utf8_lossy(&output.stderr);
    assert!(
        errors.contains("error: GPU build: array iterators support only `next`"),
        "{errors}"
    );
    assert!(errors.contains("src/main.rs:8:16"), "{errors}");
    assert!(
        errors.contains("note: in code from another crate ("),
        "{errors}"
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn installed_compiler_executables_build_kernels_after_an_identity_check() {
    let project = Project::new(&source(5));
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let state = project.state("scale", MathMode::Precise, false);
    let bin = project.0.join("bin");
    std::fs::create_dir(&bin).unwrap();
    let installed = bin.join("metal-oxide-compiler");
    std::fs::copy(state["compiler"].as_str().unwrap(), &installed).unwrap();
    let build = |compiler: &Path| {
        let mut command = project.cargo(&["build", "--quiet"]);
        command.env("METAL_OXIDE_COMPILER", compiler);
        command.output().unwrap()
    };
    checked(build(&installed));
    let state = project.state("scale", MathMode::Precise, false);
    assert_eq!(Path::new(state["compiler"].as_str().unwrap()), installed);

    // A compiler that reports another version is rejected.
    let rustc = checked(
        Command::new("rustup")
            .args([
                "run",
                metal_oxide_artifact::COMPILER_NIGHTLY,
                "rustc",
                "-vV",
            ])
            .output()
            .unwrap(),
    );
    let info = serde_json::json!({
        "protocol": 1,
        "abi": metal_oxide_artifact::ABI_VERSION,
        "version": "0.0.0-other",
        "rustc": String::from_utf8(rustc.stdout).unwrap().trim(),
    });
    let other = bin.join("other-compiler");
    std::fs::write(
        &other,
        format!("#!/usr/bin/env python3\nimport hashlib,json,pathlib\nv=json.loads({:?})\nv['binary']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()\nprint(json.dumps(v))\n", info.to_string()),
    )
    .unwrap();
    std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o755)).unwrap();
    let errors = failure(build(&other));
    assert!(errors.contains("compiler version mismatch"), "{errors}");
    let install = format!(
        "cargo install cargo-metal --version '={}' --locked",
        env!("CARGO_PKG_VERSION")
    );
    assert!(errors.contains(&install), "{errors}");
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn kernels_build_for_the_host_whatever_target_cargo_selects() {
    let project = Project::new(&source(8));
    let host =
        String::from_utf8(checked(Command::new("rustc").arg("-vV").output().unwrap()).stdout)
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix("host: ").map(str::to_owned))
            .unwrap();
    let stale = Project::target().join("compiler/debug/metal-oxide-compiler");
    let existed = stale.exists();
    if !existed {
        std::fs::create_dir_all(stale.parent().unwrap()).unwrap();
        std::fs::write(&stale, b"stale compiler at an old default path").unwrap();
    }
    checked(
        project
            .cargo(&["build", "--quiet"])
            .env("CARGO_BUILD_TARGET", &host)
            .env("RUSTFLAGS", "--cfg host_only")
            .output()
            .unwrap(),
    );
    let state = project.state("scale", MathMode::Precise, false);
    let compiler = state["compiler"].as_str().unwrap();
    assert!(
        compiler.contains(&format!("/{host}/release/")),
        "{compiler}"
    );
    assert_ne!(Path::new(compiler), stale);
    if !existed {
        std::fs::remove_file(stale).unwrap();
    }
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn parallel_kernel_builds_share_the_target_directory() {
    let workspace = Project::new("fn main() {}\n");
    let runtime = support::root().join("crates/metal-oxide");
    for (index, member) in ["first", "second", "third"].into_iter().enumerate() {
        let package = workspace.0.join(member);
        std::fs::create_dir_all(package.join("src")).unwrap();
        std::fs::write(
            package.join("Cargo.toml"),
            format!("[package]\nname = {member:?}\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[dependencies]\nmetal-oxide = {{ path = {:?} }}\n", runtime.display().to_string()),
        )
        .unwrap();
        let kernel = source(10 + index as u32).replace("fn scale", &format!("fn scale_{member}"));
        std::fs::write(package.join("src/main.rs"), kernel).unwrap();
    }
    let manifest = workspace.0.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest).unwrap().replace(
        "[workspace]\n",
        "[workspace]\nmembers = [\"first\", \"second\", \"third\"]\n",
    );
    std::fs::write(manifest, text).unwrap();
    let built = workspace.built(&mut workspace.cargo(&["build", "--quiet", "--workspace", "-j3"]));
    let artifacts = built.values().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(artifacts.len(), 3, "{built:?}");
    for artifact in artifacts {
        let manifest = std::fs::read_to_string(artifact.join("manifest.json")).unwrap();
        let manifest = metal_oxide_artifact::Manifest::from_json(&manifest).unwrap();
        let library = std::fs::read(artifact.join("kernels.metallib")).unwrap();
        manifest.verify_library(&library).unwrap();
    }
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn kernel_builds_leave_out_host_only_dependencies() {
    let project = Project::new(
        "#[metal_oxide::kernels(path = \"kernels\")]\nmod kernels {}\n\nfn main() {}\n",
    );
    project.kernel_crate(
        "#![no_std]\nuse metal_oxide_device::kernel;\n#[kernel]\npub unsafe fn empty() {}\n",
    );
    let helper = project.0.join("host-only");
    std::fs::create_dir_all(helper.join("src")).unwrap();
    std::fs::write(
        helper.join("Cargo.toml"),
        "[package]\nname = \"host-only\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(helper.join("src/lib.rs"), "#![no_std]\n").unwrap();
    std::fs::write(
        helper.join("build.rs"),
        "fn main() { panic!(\"a host-only build script ran\") }\n",
    )
    .unwrap();
    let manifest = project.0.join("kernels/Cargo.toml");
    let text = std::fs::read_to_string(&manifest).unwrap()
        + "\n[target.'cfg(not(target_env = \"metal\"))'.dependencies]\nhost-only = { path = \"../host-only\" }\n";
    std::fs::write(manifest, text).unwrap();
    checked(
        Command::new("cargo")
            .current_dir(&project.0)
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );
    let built = project.built(&mut project.cargo(&["build", "-p", "host", "--quiet"]));
    assert!(built.contains_key("kernels"), "{built:?}");
}

/// An `xcrun` first on `PATH` whose Metal compiler reports another version, as after an Xcode
/// or Metal Toolchain update; compiling passes through to the installed one.
fn updated_metal(project: &Project, command: &mut Command) {
    let shims = project.0.join("updated-shims");
    std::fs::create_dir_all(&shims).unwrap();
    std::fs::write(
        shims.join("xcrun"),
        "#!/bin/sh\nfor argument; do [ \"$argument\" = metal ] && metal=1; [ \"$argument\" = --version ] && version=1; done\nif [ -n \"$metal\" ] && [ -n \"$version\" ]; then /usr/bin/xcrun \"$@\" && echo 'Updated: test'; exit; fi\nexec /usr/bin/xcrun \"$@\"\n",
    )
    .unwrap();
    std::fs::set_permissions(shims.join("xcrun"), std::fs::Permissions::from_mode(0o755)).unwrap();
    let path = std::env::var_os("PATH").unwrap();
    let paths = std::iter::once(shims).chain(std::env::split_paths(&path));
    command.env("PATH", std::env::join_paths(paths).unwrap());
}

/// The dependency file that rustc wrote for the project's host binary.
fn host_dependencies(project: &Project) -> String {
    let project = project.0.display().to_string();
    std::fs::read_dir(Project::target().join("debug/deps"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            name.starts_with("host-") && name.ends_with(".d")
        })
        .map(|path| std::fs::read_to_string(path).unwrap())
        .find(|info| info.contains(&project))
        .unwrap()
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn metal_toolchain_updates_rebuild_reused_kernels() {
    let main = source(4);
    let project = Project::new(&main);
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let artifact = project.artifact("scale");
    let tracked = host_dependencies(&project);
    for file in ["ToolchainInfo.plist", "SDKSettings.json"] {
        assert!(tracked.contains(file), "{file} is untracked: {tracked}");
    }
    let saved = project.0.join("metal/src/main.rs/scale-precise");
    let files = ["inputs.json", "manifest.json", "kernels.metallib"];
    let before = files.map(|file| std::fs::read(saved.join(file)).unwrap());

    project.write(&format!("{main}\n// edited outside the kernel\n"));
    let mut build = project.cargo(&["build", "--quiet"]);
    updated_metal(&project, &mut build);
    checked(build.output().unwrap());
    assert_ne!(
        project.artifact("scale"),
        artifact,
        "the build reused an artifact of the previous Metal toolchain"
    );
    // The compiler that actually ran is unchanged, and so are the saved kernels.
    for (file, bytes) in files.iter().zip(&before) {
        assert_eq!(std::fs::read(saved.join(file)).unwrap(), *bytes, "{file}");
    }
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn kernel_dependencies_keep_the_features_their_package_declares() {
    let project = Project::new(
        r#"
#[metal_oxide::kernels(dependencies(kernel_helper))]
mod scaled {
    use metal_oxide::device::*;

    #[kernel]
    pub fn scale(mut out: DisjointBuffer<u32>) {
        if let Some(out) = out.get_mut(&thread_index()) {
            *out = kernel_helper::FACTOR;
        }
    }
}

fn main() {}
"#,
    );
    let helper = project.0.join("helper");
    std::fs::create_dir_all(helper.join("src")).unwrap();
    std::fs::write(
        helper.join("Cargo.toml"),
        "[package]\nname = \"helper\"\nversion = \"0.1.0\"\nedition = \"2024\"\n\n[features]\ndefault = [\"std\"]\nstd = []\nscale = []\n",
    )
    .unwrap();
    std::fs::write(
        helper.join("src/lib.rs"),
        "#![no_std]\n#[cfg(feature = \"std\")]\nextern crate std;\n#[cfg(feature = \"scale\")]\npub const FACTOR: u32 = 3;\n",
    )
    .unwrap();
    let manifest = project.0.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest).unwrap().replace(
        "[dependencies]\n",
        "[dependencies]\nkernel-helper = { package = \"helper\", path = \"helper\", default-features = false, features = [\"scale\"] }\n",
    );
    std::fs::write(manifest, text).unwrap();
    checked(
        Command::new("cargo")
            .current_dir(&project.0)
            .args(["generate-lockfile", "--offline"])
            .output()
            .unwrap(),
    );
    let built = project.built(&mut project.cargo(&["build", "--quiet"]));
    assert!(built.contains_key("scaled"), "{built:?}");
}
