use super::support;

#[path = "../fixtures/control_helpers.rs"]
mod reference;

#[test]
fn varying_match_and_loop_exits_cannot_bypass_cooperative_helpers() {
    for fixture in ["match_barrier", "match_loop_barrier", "match_early_simd"] {
        let (output, directory) = support::emit(
            &format!("crates/metal-oxide-compiler/tests/fixtures/{fixture}.rs"),
            &["-C", "overflow-checks=off"],
        );
        support::rejected(output, "uniform participation");
        assert!(!directory.join("kernels.metal").exists());
    }
}

#[test]
fn uniform_match_and_labelled_exits_preserve_cooperative_calls() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/cooperative_control.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels[0].required_block, Some([32, 1, 1]));
    assert_eq!(abi.required_features, ["simd_groups"]);
}

#[test]
fn integer_matches_execute_with_signed_narrow_and_default_cases() {
    let values = [
        i32::MIN,
        -65537,
        -257,
        -7,
        -3,
        -1,
        0,
        1,
        2,
        7,
        9,
        10,
        12,
        13,
        255,
        256,
        65535,
        65536,
        i32::MAX,
    ];
    let expected = values.map(reference::classify);
    assert_eq!(execute("control_match", "classify", &values), expected);
}

#[test]
fn loop_exits_preserve_nested_break_continue_and_early_return() {
    let values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, u32::MAX];
    let expected = values.map(reference::search);
    assert_eq!(execute("loop_exits", "bounded_search", &values), expected);
}

#[test]
fn labelled_exits_cross_two_nested_loops() {
    let values = [0, 1, 2, 3, 4, 7, u32::MAX];
    let expected = values.map(reference::nested);
    assert_eq!(execute("loop_exits", "nested_exits", &values), expected);
}

fn execute<T: std::fmt::Display>(fixture: &str, entry: &str, values: &[T]) -> Vec<u32> {
    let (output, directory) = support::emit(
        &format!("crates/metal-oxide-compiler/tests/fixtures/{fixture}.rs"),
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let input = values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let element = if fixture == "control_match" {
        "int"
    } else {
        "uint"
    };
    let main = format!(
        "{element} input[] = {{{input}}}; uint output[{}] = {{}}; uint n = {}; \
         for (uint i = 0; i < n; ++i) {entry}(input,output,n,{{i,0,0}},{{0,0,0}},{{n,1,1}},{{1,1,1}}); \
         for (uint value : output) std::cout << value << ',';",
        values.len(),
        values.len(),
    );
    support::execute_msl(&directory, &main)
        .trim_end_matches(',')
        .split(',')
        .map(|value| value.parse::<u32>().unwrap())
        .collect()
}
