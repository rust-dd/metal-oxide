use super::{INPUTS, Saved};
use crate::{
    BuiltKernels,
    configuration::Configuration,
    environment::Environment,
    recorded::{FORMAT, Recorded, Request, profile},
};
use metal_oxide_artifact::{
    ABI_VERSION, Abi, ArtifactFile, BuildInfo, Files, Kernel, Manifest, MathMode, sha256,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Directory(PathBuf);

impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-saved-{}-{}",
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

/// An artifact that a kernel build left in `root/name`, with the inputs it recorded; `build`
/// stands for the identity of the compiler and machine that built it.
fn built(root: &Path, name: &str, library: &[u8], build: &str) -> (BuiltKernels, Recorded) {
    let directory = root.join(name);
    std::fs::create_dir(&directory).unwrap();
    let abi = Abi {
        version: ABI_VERSION,
        math_mode: MathMode::Precise,
        required_features: vec![],
        kernels: vec![Kernel {
            name: "scale".into(),
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
    let fingerprint = sha256(build.as_bytes());
    let manifest = Manifest {
        abi: abi.clone(),
        checks: vec![],
        files: Files {
            msl: sha256(b"msl"),
            oxide_ir: sha256(b"oxide ir"),
            rustc_args: sha256(build.as_bytes()),
            ir: sha256(b"air"),
            metallib: sha256(library),
        },
        build: BuildInfo {
            fingerprint: fingerprint.clone(),
            compiler: build.into(),
            rustc: "nightly".into(),
            metal: "metal".into(),
            sdk: "26.5".into(),
            metal_flags,
        },
    };
    let write = |file: ArtifactFile, bytes: &[u8]| {
        std::fs::write(directory.join(file.name()), bytes).unwrap();
    };
    write(
        ArtifactFile::Manifest,
        manifest.to_json().unwrap().as_bytes(),
    );
    write(ArtifactFile::Metallib, library);
    let inputs = Recorded {
        format: FORMAT,
        artifact: fingerprint,
        request: Request {
            source: sha256(b"scale"),
            math_mode: MathMode::Precise,
            debug_info: false,
        },
        profile: profile(),
        configuration: Configuration::record(root, root, false).unwrap(),
        environment: Environment::default(),
        lock: "Cargo.lock".into(),
        packages: vec!["metal-oxide-device 0.1.0".into()],
        files: BTreeMap::new(),
    };
    let built = BuiltKernels {
        directory,
        abi,
        tracked: vec![],
        environment: vec![],
        saved: false,
    };
    (built, inputs)
}

#[test]
fn saved_artifacts_are_named_by_file_modules_item_and_math_mode() {
    let package = Path::new("/package");
    let file = Path::new("src/lib.rs");
    let top = Saved::new(package, file, &[], "vec_add", MathMode::Precise);
    assert_eq!(top.name, Path::new("metal/src/lib.rs/vec_add-precise"));
    assert_eq!(top.directory, package.join(&top.name));
    let modules = ["tests".to_owned(), "f32".to_owned()];
    let nested = Saved::new(package, file, &modules, "vec_add", MathMode::Fast);
    assert_eq!(
        nested.name,
        Path::new("metal/src/lib.rs/tests/f32/vec_add-fast")
    );
}

#[test]
fn saved_artifacts_change_with_kernels_or_inputs_but_not_the_build_identity() {
    let root = Directory::new();
    let saved = Saved::new(
        &root.0,
        Path::new("src/main.rs"),
        &[],
        "scale",
        MathMode::Precise,
    );
    let (first, inputs) = built(&root.0, "first", b"library", "first compiler");
    assert!(!saved.holds(&first, &inputs).unwrap());
    saved.write(&first, &inputs).unwrap();
    assert!(saved.holds(&first, &inputs).unwrap());

    let (rebuilt, rebuilt_inputs) = built(&root.0, "rebuilt", b"library", "second compiler");
    assert!(saved.holds(&rebuilt, &rebuilt_inputs).unwrap());

    let (changed, changed_inputs) = built(&root.0, "changed", b"changed", "third compiler");
    assert!(!saved.holds(&changed, &changed_inputs).unwrap());
    let updated = Recorded {
        packages: vec!["metal-oxide-device 0.2.0".into()],
        ..inputs
    };
    assert!(!saved.holds(&first, &updated).unwrap());
}

#[test]
fn older_saved_input_formats_are_reported_before_their_fields() {
    let root = Directory::new();
    let saved = Saved::new(
        &root.0,
        Path::new("src/main.rs"),
        &[],
        "scale",
        MathMode::Precise,
    );
    let (built, inputs) = built(&root.0, "built", b"library", "compiler");
    saved.write(&built, &inputs).unwrap();
    let path = saved.directory.join(INPUTS);
    let mut record =
        serde_json::from_slice::<serde_json::Value>(&std::fs::read(&path).unwrap()).unwrap();
    record["format"] = 3.into();
    record["configuration"]["copied"] = true.into();
    std::fs::write(&path, record.to_string()).unwrap();
    let error = saved.load(&inputs.request, None).err().unwrap().to_string();
    assert!(
        error.contains("unsupported saved input format 3"),
        "{error}"
    );
}
