#![cfg(feature = "rustc-private")]

mod support;

use support::{
    differential::{Case, Outcome, Run},
    generate::Word,
    output,
};

fn case() -> Case {
    let mut case = Case::generated(0);
    case.inputs = vec![0];
    case.words = vec![Word::Exact, Word::Exact];
    case.output = "u32".into();
    case.block = [64, 1, 1];
    case
}

#[test]
fn generated_wide_results_preserve_both_halves_in_the_oracle() {
    let (case, expected) = (0..)
        .map(Case::generated)
        .filter(|case| case.output.ends_with("64") && case.counters.is_none())
        .map(|case| {
            let expected = case.cpu().unwrap();
            (case, expected)
        })
        .find(|(_, expected)| {
            let words = |outcome: &Outcome| matches!(outcome, Outcome::Words(_));
            expected.threads.iter().all(words)
        })
        .unwrap();
    let layout = case.layout();
    assert_eq!(layout.width, 2);
    let mut words = layout.initial();
    for (thread, outcome) in expected.threads.iter().enumerate() {
        let Outcome::Words(result) = outcome else {
            unreachable!("every thread completes");
        };
        let start = thread * case.words.len();
        words[start..start + case.words.len()].copy_from_slice(result);
    }
    assert_eq!(case.gpu_mismatch(&expected, None, &words, &[]), None);
    for high in (1..case.words.len()).step_by(2) {
        words[high] ^= 1;
        assert!(case.gpu_mismatch(&expected, None, &words, &[]).is_some());
        words[high] ^= 1;
    }
}

#[test]
fn the_oracle_rejects_missing_zero_writes_and_overwritten_padding() {
    let case = case();
    let expected = Run {
        threads: vec![Outcome::Words(vec![0, 0])],
        counters: Vec::new(),
    };
    let mut words = case.layout().initial();
    assert!(case.gpu_mismatch(&expected, None, &words, &[]).is_some());
    words[..2].fill(0);
    assert_eq!(case.gpu_mismatch(&expected, None, &words, &[]), None);
    for index in [2, 127, words.len() - 1] {
        words[index] = 0;
        let error = case.gpu_mismatch(&expected, None, &words, &[]).unwrap();
        assert!(error.contains("guard word"), "{error}");
        words[index] = output::POISON;
    }
    words.pop();
    assert!(case.gpu_mismatch(&expected, None, &words, &[]).is_some());
}
