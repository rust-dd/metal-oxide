use std::path::Path;

use metal_oxide::{Device, Error, Module};
use metal_oxide_artifact::{Abi, ArtifactFile, BuildInfo, Files, Manifest, sha256};
use objc2_metal::{MTLDevice, MTLGPUFamily};

/// Loads generated kernels as an artifact: offline MSL compilation and a manifest of the outputs.
pub fn module(device: &Device, directory: &Path) -> metal_oxide::Result<Module> {
    let (manifest, metallib) = manifest(directory)?;
    Module::from_bytes(device, &manifest.to_json()?, &metallib)
}

/// Whether `device` has the GPU family that Apple's feature tables require for `feature`; the
/// Mac2-only paravirtual device of hosted macOS VMs has none of them.
pub fn supports(device: &Device, feature: &str) -> bool {
    let family = match feature {
        "bfloat16" => MTLGPUFamily::Apple6,
        "float32_atomics" => MTLGPUFamily::Apple7,
        "uint64_atomics" => MTLGPUFamily::Apple8,
        "simdgroup_matrix" => MTLGPUFamily::Apple7,
        _ => panic!("no GPU family recorded for {feature}"),
    };
    device.as_native().supportsFamily(family)
}

/// Generated kernels that need `features` as a module, or `None` once a device lacking one has
/// rejected them as unsupported.
pub fn supported_module(
    device: &Device,
    directory: &Path,
    features: &[&str],
) -> metal_oxide::Result<Option<Module>> {
    let module = module(device, directory);
    let missing = features
        .iter()
        .filter(|feature| !supports(device, feature))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return module.map(Some);
    }
    match module {
        Err(Error::UnsupportedDevice(reason))
            if missing.iter().any(|feature| reason.contains(**feature)) =>
        {
            Ok(None)
        }
        other => panic!(
            "{} lacks {missing:?}, but loading gave {:?}",
            device.name(),
            other.map(drop)
        ),
    }
}

/// The manifest and library of generated kernels, compiled offline.
pub fn manifest(directory: &Path) -> metal_oxide::Result<(Manifest, Vec<u8>)> {
    let metal_flags = metal_oxide_build::compile_msl(directory)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    let read = |file: ArtifactFile| std::fs::read(directory.join(file.name())).unwrap();
    let metal = metal_oxide_build::metal_compiler().unwrap();
    let manifest = Manifest {
        abi: Abi::from_json(&String::from_utf8(read(ArtifactFile::Abi)).unwrap())?,
        checks: serde_json::from_slice(&read(ArtifactFile::Checks)).unwrap(),
        files: Files::from_directory(directory)?,
        build: BuildInfo {
            fingerprint: sha256(&read(ArtifactFile::Msl)),
            compiler: "metal-oxide-compiler tests".into(),
            rustc: "nightly".into(),
            metal: metal.lines().next().unwrap().into(),
            sdk: "macosx".into(),
            metal_flags,
        },
    };
    Ok((manifest, read(ArtifactFile::Metallib)))
}

/// Loads kernels generated directly from IR, writing the outputs the compiler would.
pub fn ir_module(device: &Device, ir: &metal_oxide_ir::Module) -> metal_oxide::Result<Module> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let directory = super::directory().join(format!("ir-{}", NEXT.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&directory)?;
    let codegen = metal_oxide_codegen::Codegen::new(ir).unwrap();
    let abi = codegen.abi()?;
    let write =
        |file: ArtifactFile, bytes: Vec<u8>| std::fs::write(directory.join(file.name()), bytes);
    write(ArtifactFile::Msl, codegen.emit().unwrap().into_bytes())?;
    write(ArtifactFile::OxideIr, format!("{ir:?}\n").into_bytes())?;
    write(ArtifactFile::Abi, abi.to_json()?.into_bytes())?;
    write(
        ArtifactFile::Checks,
        serde_json::to_vec(codegen.checks()).unwrap(),
    )?;
    write(ArtifactFile::RustcArgs, b"[]".to_vec())?;
    module(device, &directory)
}
