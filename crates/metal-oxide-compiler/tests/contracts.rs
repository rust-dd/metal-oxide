#![cfg(feature = "rustc-private")]

mod support;

#[test]
fn required_relations_reach_the_abi_and_prove_the_unguarded_vec_add() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/requires.rs",
        &[],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    let relations = |kernel: &str| {
        abi.kernels
            .iter()
            .find(|candidate| candidate.name == kernel)
            .unwrap()
            .requires
            .iter()
            .map(|contract| contract.relation.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        relations("vec_add"),
        ["a.len() == out.len()", "b.len() == out.len()"]
    );
    assert_eq!(relations("pairs"), ["2 * n + 1 <= values.len()", "n > 0"]);
    let checks = serde_json::from_str::<Vec<metal_oxide_artifact::Check>>(
        &std::fs::read_to_string(directory.join("checks.json")).unwrap(),
    )
    .unwrap();
    // Only `pairs` keeps runtime checks: its arithmetic relation is checked at launch only.
    assert!(
        checks.iter().all(|check| check.location.line == 20),
        "{checks:?}"
    );
}

#[test]
fn invalid_requirements_are_reported_at_their_attribute() {
    let (output, _) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/requires_invalid.rs",
        &[],
    );
    let error = String::from_utf8_lossy(&output.stderr).into_owned();
    support::rejected(output, "`n.len()` requires a buffer parameter");
    assert!(error.contains("requires_invalid.rs:5:18"), "{error}");
}
