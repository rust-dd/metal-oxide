use metal_oxide_artifact::*;

fn manifest() -> Manifest {
    Manifest {
        abi: Abi {
            version: ABI_VERSION,
            kernels: vec![Kernel {
                name: "vec_add".into(),
                parameters: vec![Parameter {
                    name: "a".into(),
                    binding: 0,
                    ty: ParameterType::Buffer {
                        element: Scalar::F32,
                        access: Access::Read,
                    },
                }],
                required_block: None,
            }],
        },
        target: DEVICE_TARGET.into(),
        msl_version: MSL_VERSION.into(),
        required_features: vec![],
        files: Files {
            msl: sha256(b"msl"),
            oxide_ir: sha256(b"oxide ir"),
            ir: sha256(b"ir"),
            metallib: sha256(b"library"),
            bindings: sha256(b"bindings"),
        },
        build: BuildInfo {
            fingerprint: sha256(b"build"),
            compiler: "oxide alpha".into(),
            rustc: "nightly".into(),
            metal: "metal".into(),
            sdk: "macosx".into(),
            rust_flags: vec![],
            metal_flags: vec![],
        },
    }
}

#[test]
fn manifest_round_trip_and_library_identity() {
    let original = manifest();
    let decoded = Manifest::from_json(&original.to_json().unwrap()).unwrap();
    assert_eq!(original, decoded);
    decoded.verify_library(b"library").unwrap();
    assert!(decoded.verify_library(b"changed library").is_err());
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    for scalar in [Scalar::F32, Scalar::U32, Scalar::I32] {
        assert_eq!((scalar.size(), scalar.alignment()), (4, 4));
    }
}

#[test]
fn rejects_incompatible_metadata() {
    let mut value = manifest();
    value.abi.version += 1;
    assert!(
        value
            .validate()
            .unwrap_err()
            .to_string()
            .contains("ABI version")
    );
    let mut value = manifest();
    value.required_features.push("atomic_u64".into());
    assert!(value.validate().is_err());
    let mut value = manifest();
    value.msl_version = "4.0".into();
    assert!(value.validate().is_err());
    let mut value = manifest();
    value.build.fingerprint = "unversioned".into();
    assert!(value.validate().is_err());
    let mut value = manifest();
    value.abi.kernels[0].required_block = Some([256, 0, 1]);
    assert!(value.validate().is_err());
}

#[test]
fn rejects_conflicting_names_slots_and_unknown_layouts() {
    let mut value = manifest();
    value.abi.kernels.push(value.abi.kernels[0].clone());
    assert!(value.validate().is_err());
    let mut value = manifest();
    value.abi.kernels[0].parameters[0].binding = 1;
    assert!(value.validate().is_err());
    let value = manifest().to_json().unwrap();
    assert!(Manifest::from_json(&value.replace("\"f32\"", "\"f64\"")).is_err());
    assert!(
        Manifest::from_json(&value.replace("\"binding\": 0", "\"binding\": 0, \"offset\": 8"))
            .is_err()
    );
    assert!(
        Manifest::from_json(&value.replace("\"version\": 1", "\"version\": 1, \"ignored\": true"))
            .is_err()
    );
}
