#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod support;

use std::{os::unix::fs::PermissionsExt, process::Command};
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

/// A `rustup` first on `PATH` that lists no compiler nightly, as on a machine without the
/// kernel toolchain.
fn without_toolchain(project: &Project, command: &mut Command) {
    let path = std::env::var_os("PATH").unwrap();
    let rustup = std::env::split_paths(&path)
        .map(|directory| directory.join("rustup"))
        .find(|candidate| candidate.is_file())
        .unwrap();
    let shims = project.0.join("shims");
    std::fs::create_dir_all(&shims).unwrap();
    let shim = shims.join("rustup");
    std::fs::write(
        &shim,
        format!(
            "#!/bin/sh\nif [ \"$1 $2\" = \"toolchain list\" ]; then echo stable; exit 0; fi\nexec {:?} \"$@\"\n",
            rustup.display().to_string()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    let paths = std::iter::once(shims).chain(std::env::split_paths(&path));
    command.env("PATH", std::env::join_paths(paths).unwrap());
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn saved_artifacts_build_unchanged_kernels_without_the_toolchain() {
    let project = Project::new(&source(2));
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    let saved = project.0.join("metal/src/main.rs/scale-precise");
    let library = std::fs::read(saved.join("kernels.metallib")).unwrap();
    assert_eq!(
        library,
        std::fs::read(project.artifact("scale").join("kernels.metallib")).unwrap()
    );

    let target = project.0.join("elsewhere");
    let build = || {
        let mut command = project.cargo(&["build", "--quiet"]);
        command.env("CARGO_TARGET_DIR", &target);
        without_toolchain(&project, &mut command);
        command
    };
    checked(build().output().unwrap());
    assert!(!target.join("metal").exists());

    project.write(&source(3));
    let errors = failure(build().output().unwrap());
    assert!(
        errors.contains("metal/src/main.rs/scale-precise is stale"),
        "{errors}"
    );
    assert!(errors.contains("needs the toolchain"), "{errors}");
    let locked = || {
        let mut command = project.cargo(&["build", "--quiet"]);
        command.env("METAL_OXIDE_ARTIFACTS", "locked");
        command
    };
    let errors = failure(locked().output().unwrap());
    assert!(errors.contains("uses saved artifacts only"), "{errors}");

    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    checked(locked().output().unwrap());
    std::fs::write(saved.join("kernels.metallib"), b"damaged").unwrap();
    let errors = failure(locked().output().unwrap());
    assert!(errors.contains("is damaged"), "{errors}");
    checked(project.cargo(&["build", "--quiet"]).output().unwrap());
    assert_ne!(
        std::fs::read(saved.join("kernels.metallib")).unwrap(),
        b"damaged"
    );
}

#[test]
#[ignore = "requires the compiler nightly, rust-src and the Metal toolchain"]
fn fetched_packages_use_their_saved_artifacts() {
    let library = Project::new("fn main() {}\n");
    let write = |factor| {
        let text = source(factor).replace("fn main() {}", "");
        std::fs::write(library.0.join("src/lib.rs"), text).unwrap();
    };
    std::fs::remove_file(library.0.join("src/main.rs")).unwrap();
    write(2);
    checked(library.cargo(&["build", "--quiet"]).output().unwrap());
    let saved = library
        .0
        .join("metal/src/lib.rs/scale-precise/kernels.metallib");
    let bytes = std::fs::read(&saved).unwrap();

    // Cargo marks the packages it unpacks from a registry or a git checkout.
    std::fs::write(library.0.join(".cargo-ok"), "{\"v\":1}").unwrap();
    let application = Project::new("fn main() {}\n");
    let manifest = application.0.join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest).unwrap().replace(
        "[dependencies]\n",
        &format!(
            "[dependencies]\nhost-kernels = {{ package = \"host\", path = {:?} }}\n",
            library.0.display().to_string()
        ),
    );
    std::fs::write(
        &manifest,
        text.replace("name = \"host\"", "name = \"application\""),
    )
    .unwrap();
    let target = application.0.join("target");
    let build = || {
        let mut command = application.cargo(&["build", "--quiet"]);
        command.env("CARGO_TARGET_DIR", &target);
        command
    };
    checked(build().output().unwrap());
    assert!(!target.join("metal").exists());
    assert_eq!(std::fs::read(&saved).unwrap(), bytes);

    write(3);
    let errors = failure(build().output().unwrap());
    assert!(errors.contains("a published package carries"), "{errors}");
}
