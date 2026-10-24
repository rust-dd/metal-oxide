#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn repeated_casts_keep_guarded_slice_loads_infallible() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/cast_bounds.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    let checks = serde_json::from_slice::<serde_json::Value>(
        &std::fs::read(directory.join("checks.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(checks, serde_json::json!([]));
}

#[test]
fn changed_indices_and_different_casts_keep_their_checks() {
    for configuration in [
        "reassign",
        "call_mutation",
        "join",
        "loop_mutation",
        "stale_guard",
        "narrowing",
    ] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/cast_bounds_bad.rs",
            &["--cfg", configuration, "-C", "overflow-checks=off"],
        );
        support::checked(output);
        let checks = serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(directory.join("checks.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            checks.as_array().unwrap().len(),
            1,
            "{configuration}: {checks}"
        );
        assert_eq!(checks[0]["kind"], "bounds", "{configuration}: {checks}");
        let index = if configuration == "join" { 1 } else { 0 };
        let other = if configuration == "narrowing" { 256 } else { 4 };
        assert_eq!(
            support::execute_msl(
                &directory,
                &format!(
                    "uint input[] = {{11, 22, 33, 44}}; uint output[4] = {{99, 99, 99, 99}}; ulong lengths[] = {{4, 4}}; bad(input, output, {other}u, {{{index}, 0, 0}}, {{}}, {{4, 1, 1}}, {{1, 1, 1}}, metal_oxide_gate, metal_oxide_failure, lengths); std::cout << atomic_load_explicit(metal_oxide_failure, memory_order_relaxed);"
                )
            ),
            "2",
            "{configuration}"
        );
    }
}

#[test]
fn a_changed_index_still_rejects_a_varying_check_before_a_barrier() {
    let (output, _) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/cast_bounds_bad.rs",
        &[
            "--cfg",
            "join",
            "--cfg",
            "barrier",
            "-C",
            "overflow-checks=off",
        ],
    );
    support::rejected(
        output,
        "a check before a barrier or SIMD operation must be proved or uniform",
    );
}
