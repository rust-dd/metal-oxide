use metal_oxide_artifact::CompilerInfo;

#[test]
fn installed_compiler_identity_rejects_every_incompatible_component() {
    let expected = "rustc nightly commit and host";
    let info = CompilerInfo::new(expected, b"compiler");
    let json = info.to_json().unwrap();
    CompilerInfo::from_json(&json)
        .unwrap()
        .verify(expected, b"compiler")
        .unwrap();
    for field in ["protocol", "abi", "version", "rustc", "binary"] {
        let mut value = serde_json::to_value(&info).unwrap();
        value[field] = if matches!(field, "protocol" | "abi") {
            999.into()
        } else {
            "mismatch".into()
        };
        let changed = CompilerInfo::from_json(&value.to_string()).unwrap();
        assert!(
            changed.verify(expected, b"compiler").is_err(),
            "accepted {field}"
        );
    }
    assert!(info.verify(expected, b"other compiler").is_err());
    assert!(CompilerInfo::from_json("{}").is_err());
}
