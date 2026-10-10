use super::support;

#[test]
fn dynamic_shared_arguments_have_independent_slots_and_typed_counts() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/dynamic_shared.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.required_features, ["dynamic_threadgroup_memory"]);
    let parameters = &abi.kernels[0].parameters;
    assert_eq!(
        parameters.iter().map(|p| p.binding).collect::<Vec<_>>(),
        [0, 0, 1, 2, 1, 3]
    );
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    assert!(msl.contains("[[threadgroup(0)]]"));
    assert!(msl.contains("[[threadgroup(1)]]"));
    assert!(msl.contains("[[buffer(3)]]"));
    let bindings = std::fs::read_to_string(directory.join("bindings.rs")).unwrap();
    assert!(bindings.contains("Argument::threadgroup::<f32>"));
    assert!(bindings.contains("Argument::threadgroup::<(u8, u64,)>"));
    assert!(bindings.contains("r#scratch: usize"));
}

#[test]
fn dynamic_shared_helpers_preserve_uniform_participation() {
    let (output, _) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/dynamic_divergent.rs",
        &["-C", "overflow-checks=off"],
    );
    support::rejected(output, "uniform");
}
