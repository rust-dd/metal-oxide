#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn rust_nested_values_copies_and_generic_array_helpers_execute() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/aggregates.rs",
        &[],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "float out[5] = {}; float seed = 10.0; aggregates(out, seed, {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate); for (float x : out) std::cout << x << ' ';",
        ),
        "10 11 13 11 10 ",
    );
}

#[test]
fn owned_constants_use_rust_field_layout_and_preserve_nested_values() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/aggregate_constants.rs",
        &[],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "float out[4] = {}; float seed = 9.0; aggregate_constants(out, seed, {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate); for (float x : out) std::cout << x << ' ';"
        ),
        "7 3 9 4 "
    );
}

#[test]
fn nested_updates_preserve_copies_and_loop_branch_returns() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/aggregate_flow.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "float seed = 10.0; for (uint mode : {0u, 1u, 2u, 5u}) { float out[5] = {}; aggregate_flow(out, seed, mode, {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate); for (float x : out) std::cout << x << ' '; }"
        ),
        "10 11 12 13 13 12 11 12 23 13 11 11 12 23 13 7 11 12 53 13 "
    );
}

#[test]
fn unproved_dynamic_array_bounds_are_checked_at_runtime_in_both_overflow_modes() {
    for options in [&[][..], &["-C", "overflow-checks=off"][..]] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/aggregate_dynamic_index.rs",
            options,
        );
        support::checked(output);
        let checks = serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(directory.join("checks.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(checks[0]["kind"], "bounds", "{checks}");
        assert_eq!(checks[0]["location"]["line"], 7, "{checks}");
        assert!(
            checks[0]["location"]["file"]
                .as_str()
                .unwrap()
                .ends_with("tests/fixtures/aggregate_dynamic_index.rs"),
            "{checks}"
        );
        // An out-of-bounds thread records its check, then returns before its store.
        assert_eq!(
            support::execute_msl(
                &directory,
                "float out[1] = {-1}; for (uint index : {2u, 3u}) { out[0] = -1; dynamic_index(out, index, 5.0f, {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate, metal_oxide_failure); std::cout << out[0] << ',' << atomic_load_explicit(metal_oxide_failure, memory_order_relaxed) << ';'; }",
            ),
            "5,0;-1,2;"
        );
    }
}

#[test]
fn uninitialized_private_arrays_emit_no_initializer() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/uninit_array.rs",
        &[],
    );
    support::checked(output);
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    assert!(!msl.contains("{{"), "{msl}");
    assert_eq!(
        support::execute_msl(
            &directory,
            "float out[6] = {}; float input[5] = {1, 2, 3, 4, 5}; uint count = 5; running_sums(out, input, count, {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate); for (float x : out) std::cout << x << ' ';",
        ),
        "1 3 6 10 15 0 ",
    );
}

#[test]
fn zero_length_owned_arrays_have_an_explicit_diagnostic() {
    let (output, _) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/aggregate_empty.rs",
        &[],
    );
    support::rejected(output, "array length must be a nonzero concrete");
}

#[test]
fn structured_parameters_and_record_buffers_emit_a_shared_host_contract() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/structured.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels[0].parameters.len(), 4);
}
