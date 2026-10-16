use super::support;

#[test]
fn lifetime_parameters_are_checked_and_erased_without_instances() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/lifetimes.rs",
        &[],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels.len(), 3);
    for name in ["late", "early", "generic_u32"] {
        assert!(abi.kernels.iter().any(|entry| entry.name == name));
    }
    for (case, error) in [
        ("escape", "cannot return reference to local variable"),
        ("outlives", "lifetime may not live long enough"),
    ] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/rejections/lifetimes.rs",
            &["--cfg", case],
        );
        support::rejected(output, error);
        assert!(!directory.join("abi.json").exists());
    }
}

#[test]
fn invalid_generic_instances_fail_before_an_artifact_is_written() {
    for (case, expected) in [
        ("empty", "at least one instance"),
        ("duplicate", "duplicate kernel instance name"),
        ("nongeneric", "generic arguments"),
        ("arity", "generic arguments"),
        ("bound", "no implementation for `f32 & f32`"),
        ("unsupported", "unsupported kernel parameter type"),
        ("collision", "duplicate kernel name"),
        ("calls_entry", "calling a kernel entrypoint is unsupported"),
    ] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/rejections/generic_instances.rs",
            &["--cfg", case],
        );
        support::rejected(output, expected);
        assert!(!directory.join("kernels.metal").exists(), "{case}");
        assert!(!directory.join("abi.json").exists(), "{case}");
    }
}
