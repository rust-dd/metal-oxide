#![cfg(feature = "rustc-private")]

mod support;

fn emit(fixture: &str) -> std::path::PathBuf {
    let (output, directory) = support::emit(
        &format!("crates/metal-oxide-compiler/tests/fixtures/{fixture}.rs"),
        &[],
    );
    support::checked(output);
    directory
}

#[test]
fn a_mutable_reference_to_a_local_writes_through() {
    let directory = emit("reference_use");
    assert_eq!(
        support::execute_msl(
            &directory,
            "uint out[1] = {}; reference_use(out, 41u, {}, {}, {}, {}, metal_oxide_gate, metal_oxide_failure); std::cout << out[0];",
        ),
        "42"
    );
}

#[test]
fn slices_of_local_arrays_index_subslice_and_reborrow_without_runtime_checks() {
    let directory = emit("slices");
    assert_eq!(
        support::execute_msl(
            &directory,
            "float out[8] = {}; slices(out, 1.0f, {}, {}, {}, {}, metal_oxide_gate); for (float x : out) std::cout << x << ' ';",
        ),
        "11 5 8 3 11 5 1 10 "
    );
    let checks = std::fs::read_to_string(directory.join("checks.json")).unwrap();
    assert_eq!(checks.trim(), "[]", "{checks}");
}
