use crate::{
    cache,
    cli::{Action, Options},
    metadata,
    process::{self, Result},
    rust::Rust,
    toolchain::{METAL_FLAGS, Metal},
};
use metal_oxide_artifact::{Abi, BuildInfo, DEVICE_TARGET, Files, MSL_VERSION, Manifest, sha256};
use std::{
    path::Path,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

pub(crate) fn execute(options: &Options) -> Result<()> {
    let project = metadata::load(options)?;
    let rust = Rust::prepare(&project)?;
    let output = rust.kernels(&project)?;
    if options.action == Action::Inspect {
        print!("{}", std::fs::read_to_string(output.join("kernels.metal"))?);
        return Ok(());
    }
    let metal = Metal::find()?;
    let source = cache::sources(&project)?;
    let generated = [
        "kernels.metal",
        "kernels.oxide-ir",
        "abi.json",
        "bindings.rs",
    ]
    .map(|name| std::fs::read(output.join(name)))
    .into_iter()
    .collect::<std::io::Result<Vec<_>>>()?;
    let features = project
        .metadata
        .resolve
        .nodes
        .iter()
        .filter(|n| project.dependencies.contains(&n.id))
        .map(|n| (&n.id, &n.features))
        .collect::<Vec<_>>();
    let profile = std::env::vars()
        .filter(|(key, _)| key.starts_with("CARGO_PROFILE_RELEASE_"))
        .collect::<std::collections::BTreeMap<_, _>>();
    let identity = serde_json::to_vec(&(
        env!("CARGO_PKG_VERSION"),
        &source,
        &generated,
        &features,
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
    let root = project.metadata.target_directory.join("metal");
    let directory = root.join(&fingerprint);
    if !cache::valid(&directory, &fingerprint) {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let stage = root.join(format!(
            ".build-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&stage)?;
        let result = create(&stage, &output, &rust, &metal, &fingerprint);
        if let Err(error) = result {
            std::fs::remove_dir_all(&stage)?;
            return Err(error);
        }
        if cache::valid(&directory, &fingerprint) {
            std::fs::remove_dir_all(&stage)?;
        } else {
            if directory.exists() {
                std::fs::remove_dir_all(&directory)?;
            }
            std::fs::rename(&stage, &directory)?;
        }
        eprintln!("Built Metal artifact {}", directory.display());
    } else {
        eprintln!("Cached Metal artifact {}", directory.display());
    }
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
        .env("METAL_OXIDE_BINDINGS", directory.join("bindings.rs"))
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
    output: &Path,
    rust: &Rust,
    metal: &Metal,
    fingerprint: &str,
) -> Result<()> {
    for name in [
        "kernels.metal",
        "kernels.oxide-ir",
        "abi.json",
        "bindings.rs",
    ] {
        std::fs::copy(output.join(name), stage.join(name))?;
    }
    let abi = Abi::from_json(&std::fs::read_to_string(stage.join("abi.json"))?)?;
    metal.compile(stage)?;
    let hash = |name: &str| -> Result<String> { Ok(sha256(&std::fs::read(stage.join(name))?)) };
    let manifest = Manifest {
        required_features: abi.required_features.clone(),
        abi,
        target: DEVICE_TARGET.into(),
        msl_version: MSL_VERSION.into(),
        files: Files {
            msl: hash("kernels.metal")?,
            oxide_ir: hash("kernels.oxide-ir")?,
            ir: hash("kernels.ir")?,
            metallib: hash("kernels.metallib")?,
            bindings: hash("bindings.rs")?,
        },
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
    std::fs::write(stage.join("manifest.json"), manifest.to_json()?)?;
    Ok(())
}
