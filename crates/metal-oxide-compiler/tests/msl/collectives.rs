use super::support;
#[test]
fn typed_simd_operations_compile_and_record_capabilities() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/simd_ops.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels.len(), 5);
    assert_eq!(abi.required_features, ["simd_groups"]);
}
#[test]
fn collectives_reject_divergence_invalid_controls_and_types() {
    for (cfg, message) in [
        ("divergent", "uniform participation"),
        ("control", "uniform participation"),
        ("lane", "invalid static SIMD source"),
        ("float_bits", "integer SIMD"),
        ("narrow", "SIMD numeric"),
    ] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/invalid_simd.rs",
            &["-C", "overflow-checks=off", "--cfg", cfg],
        );
        support::rejected(output, message);
        assert!(!directory.join("kernels.metal").exists());
    }
}
