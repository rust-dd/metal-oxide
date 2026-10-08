use crate::{
    cache,
    cli::{Action, Options},
    metadata,
    process::{self, Result},
    rust::Rust,
    toolchain::{METAL_FLAGS, Metal},
};
use metal_oxide_artifact::{
    Abi, ArtifactFile, BuildInfo, COMPILER_OUTPUTS, DEVICE_TARGET, Files, MSL_VERSION, Manifest,
    sha256,
};
use std::{
    fs::{OpenOptions, TryLockError},
    path::Path,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

pub(crate) fn execute(options: &Options) -> Result<()> {
    let project = metadata::load(options)?;
    let root = project.metadata.target_directory.join("metal");
    std::fs::create_dir_all(&root)?;
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join(".lock"))?;
    match lock.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            eprintln!("Blocking waiting for Metal build lock");
            lock.lock()?;
        }
        Err(TryLockError::Error(error)) => return Err(error.into()),
    }
    cache::discard_abandoned(&root)?;
    let rust = Rust::prepare(&project)?;
    let kernels = rust.kernels(&project)?;
    let output = &kernels.output;
    if options.action == Action::Inspect {
        print!(
            "{}",
            std::fs::read_to_string(output.join(ArtifactFile::Msl.name()))?
        );
        return Ok(());
    }
    let metal = Metal::find()?;
    let generated = COMPILER_OUTPUTS
        .map(|file| std::fs::read(output.join(file.name())))
        .into_iter()
        .collect::<std::io::Result<Vec<_>>>()?;
    let profile = std::env::vars()
        .filter(|(key, _)| key.starts_with("CARGO_PROFILE_RELEASE_"))
        .collect::<std::collections::BTreeMap<_, _>>();
    let identity = serde_json::to_vec(&(
        env!("CARGO_PKG_VERSION"),
        &kernels.inputs,
        &generated,
        &rust.identity,
        &rust.version,
        &rust.flags,
        std::fs::read(&rust.target)?,
        &metal.version,
        &metal.sdk,
        METAL_FLAGS,
        profile,
    ))?;
    let fingerprint = sha256(&identity);
    let directory = root.join(&fingerprint);
    if !cache::valid(&directory, &fingerprint) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let stage = root.join(format!(
            ".build-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&stage)?;
        let result = create(&stage, &generated, &rust, &metal, &fingerprint);
        if let Err(error) = result {
            let failed = root.join(
                stage
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .replacen(".build-", ".failed-", 1),
            );
            std::fs::rename(&stage, &failed)?;
            return Err(format!(
                "Metal build failed; inputs retained at {}: {error}",
                failed.display()
            )
            .into());
        }
        if directory.exists() {
            std::fs::remove_dir_all(&directory)?;
        }
        std::fs::rename(&stage, &directory)?;
        eprintln!("Built Metal artifact {}", directory.display());
    } else {
        eprintln!("Cached Metal artifact {}", directory.display());
    }
    drop(lock);
    let mut host = Command::new("cargo");
    host.current_dir(&project.metadata.workspace_root)
        .arg(match options.action {
            Action::Build => "build",
            Action::Run => "run",
            Action::Test => "test",
            Action::Inspect => unreachable!(),
        })
        .args(["--package", &project.host().name, "--locked"])
        .arg("--manifest-path")
        .arg(&project.host().manifest_path)
        .env(
            "METAL_OXIDE_BINDINGS",
            directory.join(ArtifactFile::Bindings.name()),
        )
        .env("METAL_OXIDE_ARTIFACT_DIR", &directory);
    if options.release {
        host.arg("--release");
    }
    if options.action == Action::Test {
        host.args(["--", "--ignored", "--test-threads=1"]);
    } else if !options.arguments.is_empty() {
        host.arg("--");
    }
    host.args(&options.arguments);
    process::run(&mut host)
}

fn create(
    stage: &Path,
    generated: &[Vec<u8>],
    rust: &Rust,
    metal: &Metal,
    fingerprint: &str,
) -> Result<()> {
    for (file, bytes) in COMPILER_OUTPUTS.into_iter().zip(generated) {
        std::fs::write(stage.join(file.name()), bytes)?;
    }
    let abi = Abi::from_json(&std::fs::read_to_string(
        stage.join(ArtifactFile::Abi.name()),
    )?)?;
    metal.compile(stage)?;
    let manifest = Manifest {
        required_features: abi.required_features.clone(),
        abi,
        target: DEVICE_TARGET.into(),
        msl_version: MSL_VERSION.into(),
        files: Files::from_directory(stage)?,
        build: BuildInfo {
            fingerprint: fingerprint.into(),
            compiler: rust.identity.clone(),
            rustc: rust.version.clone(),
            metal: metal.version.clone(),
            sdk: metal.sdk.clone(),
            rust_flags: std::iter::once("--release".to_owned())
                .chain(rust.flags.iter().cloned())
                .collect(),
            metal_flags: METAL_FLAGS.iter().map(|v| (*v).to_owned()).collect(),
        },
    };
    std::fs::write(
        stage.join(ArtifactFile::Manifest.name()),
        manifest.to_json()?,
    )?;
    Ok(())
}
