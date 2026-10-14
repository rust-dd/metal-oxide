#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn repeated_helper_dag_preserves_checked_results() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_dag.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "for (uint input : {0u,1u,2u}) { uint value = 99; proof_dag(&value,input,{},{},{},{},metal_oxide_gate); std::cout << value << ','; }"
        ),
        "0,65536,99,"
    );
}

#[test]
fn helper_proofs_keep_different_argument_ranges_separate() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_dag.rs",
        &["--cfg", "different_inputs", "-C", "overflow-checks=on"],
    );
    support::checked(output);
    // `divide(2)` is proven, but the helper is shared with `divide(divisor)`.
    let zero = support::check(&directory, serde_json::json!("division_by_zero"), 67);
    assert_eq!(
        support::execute_msl(
            &directory,
            "for (uint divisor : {4u, 0u}) { uint out[1] = {7}; proof_inputs(out, divisor, {}, {}, {}, {}, metal_oxide_gate, metal_oxide_failure); std::cout << out[0] << ',' << atomic_load_explicit(metal_oxide_failure, memory_order_relaxed) << ';'; }"
        ),
        format!("48,0;7,{};", zero + 2)
    );
}

#[test]
fn guarded_arithmetic_and_array_loops_preserve_checked_rust() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_good.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    let actual = support::execute_msl(
        &directory,
        "int out[3] = {}; guarded_division(out, -123, -7, {}, {}, {}, {}, metal_oxide_gate); for (auto value : out) std::cout << value << ','; uint scalar = 0; checked_range(&scalar, 15u, {}, {}, {}, {}, metal_oxide_gate); std::cout << scalar << ','; uint array[4] = {}; checked_array_loop(array, {}, {}, {}, {}, metal_oxide_gate); for (auto value : array) std::cout << value << ','; uint sums[64] = {}; uint count = 5; nested_reverse_sums(sums, count, {}, {}, {}, {}, metal_oxide_gate); for (int i = 0; i < 8; ++i) std::cout << sums[i] << ',';",
    );
    assert_eq!(actual, "17,-4,-29,24,11,21,31,41,1,3,6,10,15,0,0,0,");
}

/// Each configuration has a check that unknown, joined or stale facts must not remove.
fn unproved_check(configuration: &str, overflow: &str) -> serde_json::Value {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_bad.rs",
        &[
            "--cfg",
            configuration,
            "-C",
            &format!("overflow-checks={overflow}"),
        ],
    );
    support::checked(output);
    serde_json::from_slice(&std::fs::read(directory.join("checks.json")).unwrap()).unwrap()
}

#[test]
fn unknown_joined_and_stale_facts_keep_their_runtime_checks() {
    for (configuration, kind) in [
        ("join", serde_json::json!("division_by_zero")),
        ("reassign", serde_json::json!("division_by_zero")),
        ("signed", serde_json::json!({"overflow": "rem"})),
        ("index", serde_json::json!("bounds")),
        ("index_reassign", serde_json::json!("bounds")),
        ("stale_condition", serde_json::json!("bounds")),
        ("overflow", serde_json::json!({"overflow": "add"})),
        ("helper", serde_json::json!("division_by_zero")),
        ("truncated_index", serde_json::json!("bounds")),
        ("dynamic_store", serde_json::json!({"overflow": "add"})),
        ("false_and", serde_json::json!("bounds")),
        ("division_range", serde_json::json!({"overflow": "add"})),
    ] {
        let checks = unproved_check(configuration, "on");
        assert!(
            checks
                .as_array()
                .unwrap()
                .iter()
                .any(|check| check["kind"] == kind),
            "{configuration}: {checks}"
        );
    }
}

#[test]
fn signed_division_checks_are_required_in_wrapping_builds() {
    let checks = unproved_check("signed", "off");
    assert_eq!(checks.as_array().unwrap().len(), 1, "{checks}");
    assert_eq!(checks[0]["kind"], serde_json::json!({"overflow": "rem"}));
}

#[test]
fn signed_division_by_constant_operands_keeps_the_overflow_check() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_constant_division.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let dividend = support::check(&directory, serde_json::json!({"overflow": "div"}), 7);
    let divisor = support::check(&directory, serde_json::json!({"overflow": "rem"}), 12);
    assert_eq!(
        support::execute_msl(
            &directory,
            "for (int divisor : {2, -1}) { int out[1] = {7}; constant_dividend(out, divisor, {}, {}, {}, {}, metal_oxide_gate, metal_oxide_failure); std::cout << out[0] << ',' << atomic_load_explicit(metal_oxide_failure, memory_order_relaxed) << ';'; atomic_store_explicit(metal_oxide_failure, 0u, memory_order_relaxed); } \
             for (char value : {char(5), char(-128)}) { char out[1] = {7}; constant_divisor(out, value, {}, {}, {}, {}, metal_oxide_gate, metal_oxide_failure); std::cout << int(out[0]) << ',' << atomic_load_explicit(metal_oxide_failure, memory_order_relaxed) << ';'; atomic_store_explicit(metal_oxide_failure, 0u, memory_order_relaxed); }"
        ),
        format!("-1073741824,0;7,{};0,0;7,{};", dividend + 2, divisor + 2)
    );
}
