use std::path::Path;

use metal_oxide::{Device, Module};
use metal_oxide_artifact::{
    Abi, ArtifactFile, BuildInfo, DEVICE_TARGET, Files, MSL_VERSION, Manifest, sha256,
};

/// Loads generated kernels as an artifact: offline MSL compilation and a manifest of the outputs.
pub fn module(device: &Device, directory: &Path) -> metal_oxide::Result<Module> {
    let (manifest, metallib) = manifest(directory)?;
    Module::from_bytes(device, &manifest.to_json()?, &metallib)
}

/// The manifest and library of generated kernels, compiled offline.
pub fn manifest(directory: &Path) -> metal_oxide::Result<(Manifest, Vec<u8>)> {
    let metal_flags = metal_oxide_build::compile_msl(directory).unwrap();
    let read = |file: ArtifactFile| std::fs::read(directory.join(file.name())).unwrap();
    let manifest = Manifest {
        abi: Abi::from_json(&String::from_utf8(read(ArtifactFile::Abi)).unwrap())?,
        checks: serde_json::from_slice(&read(ArtifactFile::Checks)).unwrap(),
        target: DEVICE_TARGET.into(),
        msl_version: MSL_VERSION.into(),
        files: Files::from_directory(directory)?,
        build: BuildInfo {
            fingerprint: sha256(&read(ArtifactFile::Msl)),
            compiler: "metal-oxide-compiler tests".into(),
            rustc: "nightly".into(),
            metal: metal_oxide_build::metal_compiler().unwrap(),
            sdk: "macosx".into(),
            rustc_args: Vec::new(),
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
    module(device, &directory)
}
