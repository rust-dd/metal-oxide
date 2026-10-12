use std::{
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

const KERNEL: &str = "use metal_oxide_device::{WriteBuffer, kernel, thread_idx};\n\n#[kernel]\npub unsafe fn fill(out: WriteBuffer<u32>, base: u32) {\n    let i = thread_idx().x;\n    unsafe { out.store_unchecked(i, base.wrapping_add(STEP * i)) };\n}\n";

const BUILD: &str = "fn main() -> Result<(), metal_oxide_build::Error> {\n    metal_oxide_build::Artifact::new(\"../kernels/Cargo.toml\", \"artifact\").update()\n}\n";

const HOST: &str = r#"mod kernels {
    include!("../artifact/bindings.rs");
}

fn main() -> metal_oxide::Result<()> {
    let device = metal_oxide::Device::system_default()?;
    let kernels = kernels::load_bytes(
        &device,
        include_str!("../artifact/manifest.json"),
        include_bytes!("../artifact/kernels.metallib"),
    )?;
    let mut out = device.buffer_zeroed::<u32>(4)?;
    // SAFETY: one block of four threads writes the four elements of `out`.
    unsafe { kernels.fill(metal_oxide::LaunchConfig::<4>::new(metal_oxide::Dim3::x(1)), &mut out, 7)? };
    println!("{:?}", out.as_slice());
    Ok(())
}
"#;

struct Workspace(PathBuf);

impl Workspace {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = repository().join(format!(
            "target/artifact-tests/{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        for name in ["host/src", "kernels/src"] {
            std::fs::create_dir_all(path.join(name)).unwrap();
        }
        let crates = repository().join("crates");
        write(
            &path,
            "Cargo.toml",
            "[workspace]\nmembers=[\"host\",\"kernels\"]\nresolver=\"3\"\n",
        );
        write(
            &path,
            "host/Cargo.toml",
            &format!(
                "[package]\nname=\"artifact-host\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[dependencies]\nmetal-oxide={{path={:?}}}\n[build-dependencies]\nmetal-oxide-build={{path={:?}}}\n",
                crates.join("metal-oxide"),
                crates.join("metal-oxide-build")
            ),
        );
        write(&path, "host/build.rs", BUILD);
        write(&path, "host/src/main.rs", HOST);
        write(
            &path,
            "kernels/Cargo.toml",
            &format!(
                "[package]\nname=\"artifact-kernels\"\nversion=\"0.1.0\"\nedition=\"2024\"\n[dependencies]\nmetal-oxide-device={{path={:?}}}\n",
                crates.join("metal-oxide-device")
            ),
        );
        write(
            &path,
            "kernels/src/lib.rs",
            &format!("#![no_std]\nconst STEP: u32 = 1;\n{KERNEL}"),
        );
        checked(
            cargo(&path)
                .args(["generate-lockfile", "--offline"])
                .output()
                .unwrap(),
        );
        Self(path)
    }

    fn run(&self, environment: &[(&str, &str)]) -> Output {
        let mut command = cargo(&self.0.join("host"));
        command
            .args(["run", "--offline"])
            .envs(environment.iter().copied());
        command.output().unwrap()
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// A shared target directory keeps the compiler and runtime builds across test runs.
fn cargo(directory: &Path) -> Command {
    let mut command = Command::new("cargo");
    command
        .current_dir(directory)
        .env(
            "CARGO_TARGET_DIR",
            repository().join("target/artifact-tests/target"),
        )
        .env_remove("METAL_OXIDE_ARTIFACTS")
        .env_remove("METAL_OXIDE_COMPILER");
    command
}

fn write(root: &Path, path: &str, contents: &str) {
    std::fs::write(root.join(path), contents).unwrap();
}

fn checked(output: Output) -> Output {
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn modified(artifact: &Path) -> std::time::SystemTime {
    std::fs::metadata(artifact.join("manifest.json"))
        .and_then(|metadata| metadata.modified())
        .unwrap()
}

#[test]
#[ignore = "requires a Metal device, the compiler nightly with rustc-dev and the Metal toolchain"]
fn build_scripts_rebuild_changed_artifacts_and_published_copies_need_no_toolchain() {
    let workspace = Workspace::new();
    let first = checked(workspace.run(&[]));
    assert_eq!(String::from_utf8_lossy(&first.stdout), "[7, 8, 9, 10]\n");
    let artifact = workspace.0.join("host/artifact");
    let built = modified(&artifact);
    let inputs = std::fs::read_to_string(artifact.join("inputs.json")).unwrap();
    assert!(inputs.contains("\"src/lib.rs\""), "{inputs}");

    // Rerunning the build script for an unrelated change keeps the artifact.
    write(
        &workspace.0,
        "host/build.rs",
        &format!("{}// rerun\n", BUILD),
    );
    checked(workspace.run(&[]));
    assert_eq!(modified(&artifact), built);
    assert_eq!(
        std::fs::read_to_string(artifact.join("inputs.json")).unwrap(),
        inputs
    );

    write(
        &workspace.0,
        "kernels/src/lib.rs",
        &format!("#![no_std]\nconst STEP: u32 = 2;\n{KERNEL}"),
    );
    let locked = workspace.run(&[("METAL_OXIDE_ARTIFACTS", "locked")]);
    assert!(!locked.status.success());
    assert!(String::from_utf8_lossy(&locked.stderr).contains("does not match its kernel sources"));
    let changed = checked(workspace.run(&[]));
    assert!(modified(&artifact) > built);
    assert_eq!(String::from_utf8_lossy(&changed.stdout), "[7, 9, 11, 13]\n");

    // Without `../kernels`, the copy behaves like a package unpacked from a registry.
    let published = workspace.0.join("registry/host");
    std::fs::create_dir_all(published.join("src")).unwrap();
    for file in ["Cargo.toml", "build.rs", "src/main.rs"] {
        std::fs::copy(workspace.0.join("host").join(file), published.join(file)).unwrap();
    }
    std::fs::create_dir(published.join("artifact")).unwrap();
    for file in ["manifest.json", "kernels.metallib", "bindings.rs"] {
        std::fs::copy(artifact.join(file), published.join("artifact").join(file)).unwrap();
    }
    let manifest = std::fs::read_to_string(published.join("Cargo.toml")).unwrap();
    write(
        &published,
        "Cargo.toml",
        &format!("{manifest}[workspace]\n"),
    );
    let output = checked(
        cargo(&published)
            .args(["run", "--offline"])
            .env("METAL_OXIDE_COMPILER", "/nonexistent/metal-oxide-compiler")
            .output()
            .unwrap(),
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "[7, 9, 11, 13]\n");
}
