use metal_oxide_artifact::*;

/// Parameters `a`, `b` and `out` are buffers; `n` and `shift` are integers.
fn parameter(name: &str) -> Option<(usize, Operand)> {
    match name {
        "a" => Some((0, Operand::Buffer)),
        "b" => Some((1, Operand::Buffer)),
        "out" => Some((2, Operand::Buffer)),
        "n" => Some((3, Operand::Integer)),
        "shift" => Some((4, Operand::Integer)),
        _ => None,
    }
}

fn holds(relation: &str, lengths: [u64; 3], values: [i128; 2]) -> bool {
    Contract::parse(relation, parameter)
        .unwrap()
        .holds(|index| lengths[index], |index| values[index - 3])
}

#[test]
fn relations_compare_lengths_values_and_literals() {
    let contract = Contract::parse("a.len() == out.len()", parameter).unwrap();
    assert_eq!(contract.left, Term::Length(0));
    assert_eq!(contract.comparison, Comparison::Eq);
    assert_eq!(contract.right, Term::Length(2));
    assert!(holds("a.len() == out.len()", [4, 0, 4], [0, 0]));
    assert!(!holds("a.len() == out.len()", [4, 0, 5], [0, 0]));
    assert!(holds("2 * n + 1 <= a.len()", [9, 0, 0], [4, 0]));
    assert!(!holds("2 * n + 1 <= a.len()", [8, 0, 0], [4, 0]));
    assert!(holds(
        "(a.len() - shift) * 2 >= 1_000usize",
        [600, 0, 0],
        [0, 100]
    ));
    assert!(holds("shift < 0", [0, 0, 0], [0, -3]));
    assert!(holds("n != 0", [0, 0, 0], [7, 0]));
}

#[test]
fn arithmetic_that_leaves_i128_fails_the_relation() {
    let product = "a.len() * b.len() * out.len() * a.len() * b.len() > 0";
    assert!(!holds(product, [u64::MAX; 3], [0, 0]));
}

#[test]
fn malformed_relations_and_unknown_parameters_are_rejected() {
    for (relation, message) in [
        ("a.len()", "one comparison"),
        ("a.len() == out.len() == b.len()", "trailing"),
        ("n.len() == 1", "requires a buffer parameter"),
        ("a == 1", "is not an integer parameter"),
        ("missing == 1", "is not an integer parameter"),
        ("a.len() == 1.5", "unsupported relation syntax"),
        ("(a.len() == 1", "unbalanced"),
        ("4u128 == n", "unsupported relation syntax"),
    ] {
        let error = Contract::parse(relation, parameter).unwrap_err();
        assert!(error.0.contains(message), "{relation}: {error}");
    }
}
