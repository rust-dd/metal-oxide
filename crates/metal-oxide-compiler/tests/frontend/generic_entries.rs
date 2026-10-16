use super::support;

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
