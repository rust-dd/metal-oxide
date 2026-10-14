#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn core_iterator_adapters_run_their_core_implementations() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/adapters.rs",
        &[],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "float a[4] = {1, 2, 3, 4}; float b[4] = {1, 1, 1, 1}; float out[4] = {}; ulong bound[3] = {4, 4, 4}; \
             for (uint x = 0; x < 4; x++) dot(a, b, out, {x, 0, 0}, {0, 0, 0}, {4, 1, 1}, {1, 1, 1}, {x, 0, 0}, metal_oxide_gate, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "10 9 7 4 "
    );
    assert_eq!(
        support::execute_msl(
            &directory,
            "float values[5] = {1, 2, 3, 4, 5}; float out[10] = {}; ulong bound[2] = {5, 10}; \
             adapters(values, out, 2.0f, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate, metal_oxide_failure, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "40 129 40 3 2 1 9 14 5 5 "
    );
    assert_eq!(
        support::execute_msl(
            &directory,
            "float out[4] = {}; \
             mutation(out, 2.0f, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate, metal_oxide_failure); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "11.5 8.5 5.5 2.5 "
    );
}

#[test]
fn raw_pointers_offset_compare_and_subtract_within_their_slice() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/raw_pointers.rs",
        &[],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "float values[5] = {1, 2, 3, 4, 5}; float out[4] = {}; ulong bound[2] = {5, 4}; \
             raw_pointers(values, out, 7, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate, metal_oxide_failure, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "75 80 5 -2 "
    );
}

#[test]
fn inclusive_ranges_iterate_through_core() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/range_inclusive.rs",
        &[],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "uint out[1] = {}; \
             range_inclusive(out, 4, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate, metal_oxide_failure); \
             for (uint x : out) std::cout << x << ' ';",
        ),
        "10 "
    );
}
