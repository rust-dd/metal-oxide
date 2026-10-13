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

#[test]
fn buffer_slices_read_the_bound_lengths_without_runtime_checks() {
    let directory = emit("buffer_lengths");
    let abi = std::fs::read_to_string(directory.join("abi.json")).unwrap();
    assert!(abi.contains("\"lengths\": true"), "{abi}");
    assert!(abi.contains("\"one_dimensional\": false"), "{abi}");
    assert_eq!(
        support::execute_msl(
            &directory,
            "float a[3] = {1, 2, 3}; float b[2] = {10, 20}; float out[4] = {}; ulong bound[3] = {3, 2, 4}; \
             for (uint x = 0; x < 4; x++) lengths(a, b, out, {x, 0, 0}, {0, 0, 0}, {4, 1, 1}, {1, 1, 1}, metal_oxide_gate, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "11 22 0 6 "
    );
    let checks = std::fs::read_to_string(directory.join("checks.json")).unwrap();
    assert_eq!(checks.trim(), "[]", "{checks}");
}

#[test]
fn each_thread_writes_its_own_disjoint_element() {
    let directory = emit("disjoint");
    let abi = std::fs::read_to_string(directory.join("abi.json")).unwrap();
    assert!(abi.contains("\"one_dimensional\": true"), "{abi}");
    assert_eq!(
        support::execute_msl(
            &directory,
            "float a[3] = {1, 2, 3}; float b[3] = {10, 20, 30}; float out[3] = {}; ulong bound[3] = {3, 3, 3}; \
             for (uint x = 0; x < 4; x++) add(a, b, out, {x % 2, 0, 0}, {x / 2, 0, 0}, {2, 1, 1}, {2, 1, 1}, {x, 0, 0}, metal_oxide_gate, metal_oxide_failure, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "11 22 33 "
    );
}

#[test]
fn slice_iterators_visit_elements_and_chunks_without_runtime_checks() {
    let directory = emit("slice_iterators");
    assert_eq!(
        support::execute_msl(
            &directory,
            "float out[3] = {}; slice_iterators(out, 2.0f, {}, {}, {}, {}, metal_oxide_gate); for (float x : out) std::cout << x << ' ';",
        ),
        "30 56 -76 "
    );
    let checks = std::fs::read_to_string(directory.join("checks.json")).unwrap();
    assert!(
        !checks.contains("bounds") && !checks.contains("slice"),
        "{checks}"
    );
}
