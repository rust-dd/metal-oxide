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
fn an_option_built_in_a_kernel_is_matched() {
    let directory = emit("option_value");
    assert_eq!(
        support::execute_msl(
            &directory,
            "uint out[2] = {9, 9}; for (uint n : {2u, 5u}) option_value(out, n, {n % 2, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate); \
             for (uint x : out) std::cout << x << ' ';",
        ),
        "9 5 "
    );
}

#[test]
fn enums_share_payload_slots_and_match_their_variants() {
    let directory = emit("enums");
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    // Circle, Square and Rect's first field share one slot; Rect's second takes another.
    assert!(
        msl.contains("uchar f0;\n    float f1;\n    float f2;\n};"),
        "{msl}"
    );
    assert_eq!(
        support::execute_msl(
            &directory,
            "float sizes[5] = {1, 2, 3, 4, 5}; float out[5] = {}; ulong bound[2] = {5, 5}; \
             for (uint x = 0; x < 5; x++) areas(sizes, out, {x, 0, 0}, {0, 0, 0}, {5, 1, 1}, {1, 1, 1}, {x, 0, 0}, metal_oxide_gate, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "3 4 12 0 75 "
    );
    assert_eq!(
        support::execute_msl(
            &directory,
            "float values[4] = {2, 7, 1, 9}; float out[4] = {}; ulong bound[2] = {4, 4}; \
             for (uint x = 0; x < 4; x++) largest_prefix(values, out, {x, 0, 0}, {0, 0, 0}, {4, 1, 1}, {1, 1, 1}, {x, 0, 0}, metal_oxide_gate, metal_oxide_failure, bound); \
             for (float x : out) std::cout << x << ' ';",
        ),
        "2 7 7 9 "
    );
}

#[test]
fn question_marks_unwraps_and_expects_report_their_callers() {
    let directory = emit("enum_methods");
    assert_eq!(
        support::execute_msl(
            &directory,
            "uint values[4] = {4, 8, 3, 6}; uint out[4] = {}; ulong bound[2] = {4, 4}; \
             for (uint x = 0; x < 4; x++) quarters(values, out, {x, 0, 0}, {0, 0, 0}, {4, 1, 1}, {1, 1, 1}, {x, 0, 0}, metal_oxide_gate, metal_oxide_failure, bound); \
             for (uint x : out) std::cout << x << ' ';",
        ),
        "5 6 3 3 "
    );
    let checks: Vec<metal_oxide_artifact::Check> =
        serde_json::from_slice(&std::fs::read(directory.join("checks.json")).unwrap()).unwrap();
    let reported = checks
        .iter()
        .map(|check| {
            (
                check.kind.message(),
                check.location.line,
                check.location.column,
            )
        })
        .collect::<Vec<_>>();
    // `#[track_caller]` reports the method call, as Rust does.
    for expected in [
        (
            "called `Option::unwrap()` on a `None` value".to_string(),
            24,
            44,
        ),
        ("an index below the length".to_string(), 25, 44),
        (
            "called `Option::unwrap()` on a `None` value".to_string(),
            40,
            50,
        ),
    ] {
        assert!(reported.contains(&expected), "{reported:?}");
    }
}

#[test]
fn zero_sized_enums_keep_their_tag_inside_records_and_behind_references() {
    let directory = emit("zero_sized_enums");
    assert_eq!(
        support::execute_msl(
            &directory,
            "uint out[1] = {0}; zero_sized(out, 7u, {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate, metal_oxide_failure); std::cout << out[0];",
        ),
        "18"
    );
}
