#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn guarded_arithmetic_and_array_loops_preserve_checked_rust() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_good.rs",
        &["-C", "overflow-checks=on"],
    );
    support::checked(output);
    let actual = support::execute_msl(
        &directory,
        "int out[3] = {}; guarded_division(out, -123, -7, {}, {}, {}, {}); for (auto value : out) std::cout << value << ','; uint scalar = 0; checked_range(&scalar, 15u, {}, {}, {}, {}); std::cout << scalar << ','; uint array[4] = {}; checked_array_loop(array, {}, {}, {}, {}); for (auto value : array) std::cout << value << ',';",
    );
    assert_eq!(actual, "17,-4,-29,24,11,21,31,41,");
}

#[test]
fn proofs_reject_unknown_joined_and_stale_facts() {
    for configuration in [
        "join",
        "reassign",
        "signed",
        "index",
        "index_reassign",
        "stale_condition",
        "overflow",
        "helper",
        "truncated_index",
        "dynamic_store",
        "false_and",
        "division_range",
    ] {
        let (output, directory) = support::emit(
            "crates/metal-oxide-compiler/tests/fixtures/proofs_bad.rs",
            &["--cfg", configuration, "-C", "overflow-checks=on"],
        );
        support::rejected(output, "MIR assertion");
        assert!(!directory.join("kernels.metal").exists(), "{configuration}");
    }
}

#[test]
fn signed_division_checks_are_required_in_wrapping_builds() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/proofs_bad.rs",
        &["--cfg", "signed", "-C", "overflow-checks=off"],
    );
    support::rejected(output, "MIR assertion");
    assert!(!directory.join("kernels.metal").exists());
}
