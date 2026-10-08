use metal_oxide_artifact::*;

fn manifest() -> Manifest {
    Manifest {
        abi: Abi {
            version: ABI_VERSION,
            required_features: vec![],
            kernels: vec![Kernel {
                name: "vec_add".into(),
                parameters: vec![Parameter {
                    name: "a".into(),
                    binding: 0,
                    ty: ParameterType::Buffer {
                        element: Layout::scalar(Scalar::F32),
                        stride: 4,
                        access: Access::Read,
                    },
                }],
                required_block: None,
            }],
        },
        target: DEVICE_TARGET.into(),
        msl_version: MSL_VERSION.into(),
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
            rustc_args: vec![],
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
fn manifest_mutations_fail_without_panicking() {
    let valid = serde_json::to_value(manifest()).unwrap();
    for (path, replacement) in [
        ("/abi/version", serde_json::json!(999)),
        ("/abi/kernels/0/parameters/0/binding", serde_json::json!(31)),
        (
            "/abi/kernels/0/parameters/0/ty/stride",
            serde_json::json!(0),
        ),
        (
            "/abi/kernels/0/parameters/0/ty/element/size",
            serde_json::json!(u64::MAX),
        ),
        (
            "/abi/kernels/0/parameters/0/ty/element/alignment",
            serde_json::json!(3),
        ),
        (
            "/abi/kernels/0/parameters/0/ty/access",
            serde_json::json!("atomic"),
        ),
        (
            "/abi/kernels/0/required_block",
            serde_json::json!([256, 0, 1]),
        ),
        ("/abi/required_features", serde_json::json!(["unknown"])),
        ("/files/metallib", serde_json::json!("invalid")),
        ("/build/fingerprint", serde_json::json!("")),
    ] {
        let mut value = valid.clone();
        *value.pointer_mut(path).unwrap() = replacement;
        assert!(
            Manifest::from_json(&value.to_string()).is_err(),
            "accepted {path}"
        );
    }
    let json = manifest().to_json().unwrap();
    for end in (0..json.len()).step_by(31) {
        assert!(Manifest::from_json(&json[..end]).is_err());
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
    value.abi.required_features.push("atomic_u64".into());
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
    let mut value = serde_json::to_value(manifest()).unwrap();
    value["abi"]["ignored"] = serde_json::json!(true);
    assert!(Manifest::from_json(&value.to_string()).is_err());
}

#[test]
fn structured_layouts_stride_and_old_abi_are_checked() {
    let mut manifest = manifest();
    let layout = Layout::record(
        "Particle",
        vec![
            ("tag".into(), Layout::scalar(Scalar::U8)),
            ("id".into(), Layout::scalar(Scalar::U32)),
        ],
    )
    .unwrap();
    manifest.abi.kernels[0].parameters[0].ty = ParameterType::Buffer {
        stride: layout.size,
        element: layout,
        access: Access::Write,
    };
    let json = manifest.to_json().unwrap();
    assert_eq!(Manifest::from_json(&json).unwrap(), manifest);
    for mutation in [
        json.replace("\"stride\": 8", "\"stride\": 4"),
        json.replace("\"offset\": 4", "\"offset\": 1"),
        json.replace("\"write\"", "\"atomic\""),
    ] {
        assert!(Manifest::from_json(&mutation).is_err(), "{mutation}");
    }
    let error =
        Manifest::from_json("{\"abi\": {\"version\": 2, \"kernels\": [{\"kind\": \"scalar\"}]}}")
            .unwrap_err();
    assert!(error.to_string().contains("ABI version 2"));
}
