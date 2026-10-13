use metal_oxide_artifact::*;

fn manifest() -> Manifest {
    Manifest {
        abi: Abi {
            version: ABI_VERSION,
            math_mode: MathMode::Precise,
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
                gate_binding: 1,
                fallible: true,
            }],
        },
        checks: vec![Check {
            kind: CheckKind::Bounds,
            location: Location {
                file: "kernels/src/lib.rs".into(),
                line: 7,
                column: 13,
            },
        }],
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
            metal_flags: MathMode::Precise.metal_flags().map(str::to_owned).to_vec(),
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
fn kernels_share_type_and_signature_tables() {
    let mut manifest = manifest();
    let args = Layout::record(
        "Args",
        vec![
            ("count".into(), Layout::scalar(Scalar::U32)),
            ("scale".into(), Layout::scalar(Scalar::F32)),
        ],
    )
    .unwrap();
    let parameters = vec![
        manifest.abi.kernels[0].parameters[0].clone(),
        Parameter {
            name: "args".into(),
            binding: 1,
            ty: ParameterType::Value { layout: args },
        },
    ];
    manifest.abi.kernels = ["first", "second", "third"]
        .map(|name| Kernel {
            name: name.into(),
            parameters: parameters.clone(),
            required_block: None,
            gate_binding: 2,
            fallible: false,
        })
        .into();
    let json = manifest.to_json().unwrap();
    assert_eq!(json.matches("\"Args\"").count(), 1);
    assert_eq!(json.matches("\"count\"").count(), 1);
    assert_eq!(Manifest::from_json(&json).unwrap(), manifest);
    let value = serde_json::from_str::<serde_json::Value>(&json).unwrap();
    assert_eq!(value["abi"]["types"].as_array().unwrap().len(), 3);
    assert_eq!(value["abi"]["signatures"].as_array().unwrap().len(), 1);
    for (pointer, reference) in [
        ("/abi/kernels/2/signature", 1),
        ("/abi/signatures/0/1/ty/layout", 3),
        ("/abi/types/2/kind/fields/0/layout", 2),
        ("/abi/types/2/kind/fields/1/layout", 3),
    ] {
        let mut invalid = value.clone();
        *invalid.pointer_mut(pointer).unwrap() = serde_json::json!(reference);
        assert!(
            Manifest::from_json(&invalid.to_string()).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn numerical_contract_requires_a_known_mode_and_matching_flags() {
    let mut fast = manifest();
    fast.abi.math_mode = MathMode::Fast;
    assert!(fast.validate().is_err());
    fast.build.metal_flags = MathMode::Fast.metal_flags().map(str::to_owned).to_vec();
    assert_eq!(Manifest::from_json(&fast.to_json().unwrap()).unwrap(), fast);
    assert_ne!(
        fast.abi.to_json().unwrap(),
        manifest().abi.to_json().unwrap()
    );
    for flags in [
        vec![],
        vec![
            "-fmetal-math-mode=fast",
            "-fmetal-math-fp32-functions=fast",
            "-ffp-contract=fast",
        ],
        vec![
            "-fmetal-math-mode=relaxed",
            "-fmetal-math-mode=safe",
            "-fmetal-math-fp32-functions=fast",
            "-ffp-contract=fast",
        ],
    ] {
        let mut invalid = fast.clone();
        invalid.build.metal_flags = flags.into_iter().map(str::to_owned).collect();
        assert!(invalid.validate().is_err());
    }
    for value in [
        serde_json::Value::Null,
        serde_json::json!("relaxed"),
        serde_json::json!(true),
    ] {
        let mut invalid = serde_json::to_value(&fast).unwrap();
        invalid["abi"]["math_mode"] = value;
        assert!(Manifest::from_json(&invalid.to_string()).is_err());
    }
    let mut legacy = serde_json::to_value(&fast).unwrap();
    legacy["abi"].as_object_mut().unwrap().remove("math_mode");
    assert!(Manifest::from_json(&legacy.to_string()).is_err());
    fast.build.metal_flags.push("-ffast-math".into());
    assert!(fast.validate().is_err());
}

#[test]
fn manifest_mutations_fail_without_panicking() {
    let valid = serde_json::to_value(manifest()).unwrap();
    for (path, replacement) in [
        ("/abi/version", serde_json::json!(999)),
        ("/abi/signatures/0/0/binding", serde_json::json!(31)),
        ("/abi/signatures/0/0/ty/stride", serde_json::json!(0)),
        ("/abi/types/0/size", serde_json::json!(u64::MAX)),
        ("/abi/types/0/alignment", serde_json::json!(3)),
        ("/abi/signatures/0/0/ty/access", serde_json::json!("atomic")),
        (
            "/abi/kernels/0/required_block",
            serde_json::json!([256, 0, 1]),
        ),
        ("/abi/required_features", serde_json::json!(["unknown"])),
        ("/abi/kernels/0/gate_binding", serde_json::json!(0)),
        ("/abi/kernels/0/gate_binding", serde_json::json!(31)),
        ("/checks/0/location/line", serde_json::json!(0)),
        ("/checks/0/kind", serde_json::json!("unknown")),
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
    let newer = json.replacen("\"version\": 4", "\"version\": 5", 1);
    assert_ne!(newer, json);
    let error = Manifest::from_json(&newer).unwrap_err();
    assert!(error.to_string().contains("ABI version 5"));
    let abi = manifest.abi.to_json().unwrap();
    let error = Abi::from_json(&abi.replacen("\"version\": 4", "\"version\": 5", 1)).unwrap_err();
    assert!(error.to_string().contains("ABI version 5"));
}

#[test]
fn check_messages_follow_rust_runtime_panics() {
    let manifest = manifest();
    let json = manifest.to_json().unwrap();
    assert!(json.contains("\"kind\": \"bounds\""), "{json}");
    assert_eq!(Manifest::from_json(&json).unwrap(), manifest);
    for (kind, message) in [
        (CheckKind::Bounds, "index out of bounds"),
        (
            CheckKind::Overflow(OverflowOp::Add),
            "attempt to add with overflow",
        ),
        (
            CheckKind::Overflow(OverflowOp::Rem),
            "attempt to calculate the remainder with overflow",
        ),
        (
            CheckKind::Overflow(OverflowOp::Shl),
            "attempt to shift left with overflow",
        ),
        (CheckKind::DivisionByZero, "attempt to divide by zero"),
        (
            CheckKind::RemainderByZero,
            "attempt to calculate the remainder with a divisor of zero",
        ),
        (
            CheckKind::Panic("assertion failed: x < n".into()),
            "assertion failed: x < n",
        ),
    ] {
        assert_eq!(kind.message(), message);
    }
}
