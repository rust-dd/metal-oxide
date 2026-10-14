#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn closures_capture_by_reference_and_by_value_and_pass_to_generic_helpers() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/closures.rs",
        &[],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            "float values[3] = {1, 2, 3}; float out[3] = {}; float factor = 2; ulong bound[2] = {3, 3}; \
             for (uint x = 0; x < 3; x++) closures(values, out, factor, {x, 0, 0}, {0, 0, 0}, {3, 1, 1}, {1, 1, 1}, {x, 0, 0}, metal_oxide_gate, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "5 8 11 "
    );
    assert_eq!(
        support::execute_msl(
            &directory,
            "float values[3] = {1, 2, 3}; float out[3] = {}; ulong bound[2] = {3, 3}; \
             for (uint x = 0; x < 3; x++) accumulate(values, out, {x, 0, 0}, {0, 0, 0}, {3, 1, 1}, {1, 1, 1}, {x, 0, 0}, metal_oxide_gate, metal_oxide_failure, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "8 9 7 "
    );
}
