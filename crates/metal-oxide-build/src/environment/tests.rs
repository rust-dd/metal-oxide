use super::*;

#[test]
fn environment_dependencies_hash_escaped_values_and_distinguish_absence() {
    let mut environment = Environment::default();
    environment
        .record(
            "# env-dep:KERNEL_UNSET\n# env-dep:KERNEL_VALUE=secret\\nline\\r\\\\path\n",
            &BTreeMap::new(),
        )
        .unwrap();
    assert_eq!(environment.0["KERNEL_UNSET"], None);
    assert_eq!(
        environment.0["KERNEL_VALUE"],
        Some(sha256(b"secret\nline\r\\path"))
    );
    assert!(
        !serde_json::to_string(&environment)
            .unwrap()
            .contains("secret")
    );
    assert_eq!(environment.names(), ["KERNEL_UNSET", "KERNEL_VALUE"]);
}

#[test]
fn generated_and_forced_values_are_not_compared_with_the_invoking_crate() {
    let mut environment = Environment::default();
    environment.record("# env-dep:CARGO_PKG_NAME=device-package\n# env-dep:CARGO_CRATE_NAME=device_package\n# env-dep:CARGO_MANIFEST_DIR=/device/package\n# env-dep:RUSTFLAGS\n# env-dep:METAL_OXIDE_CARGO_OUTPUT=/device/output\n# env-dep:FORCED_VALUE=from-configuration\n# env-dep:CARGO_CUSTOM_VALUE=ambient\n", &BTreeMap::from([("FORCED_VALUE".into(), true)])).unwrap();
    assert_eq!(environment.names(), ["CARGO_CUSTOM_VALUE"]);
}

#[test]
fn conflicting_package_values_and_invalid_escapes_are_errors() {
    let mut environment = Environment::default();
    assert!(
        environment
            .record(
                "# env-dep:VALUE=one\n# env-dep:VALUE=two\n",
                &BTreeMap::new()
            )
            .is_err()
    );
    assert!(
        Environment::default()
            .record("# env-dep:VALUE=bad\\escape\n", &BTreeMap::new())
            .is_err()
    );
}

#[test]
fn changed_ambient_values_are_stale() {
    let name = format!("METAL_OXIDE_TEST_DEPINFO_{}", std::process::id());
    let mut environment = Environment::default();
    environment
        .record(
            &format!("# env-dep:{name}=a-value-that-is-not-set\n"),
            &BTreeMap::new(),
        )
        .unwrap();
    assert!(!environment.current());
    let environment = Environment(BTreeMap::from([(name, None)]));
    assert!(environment.current());
}
