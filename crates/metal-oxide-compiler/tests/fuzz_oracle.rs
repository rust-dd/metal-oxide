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
    case
}

#[test]
fn generated_wide_results_preserve_both_halves_in_the_oracle() {
    let case = Case::generated(711);
    let expected = case.cpu().unwrap();
    assert_eq!(case.words.len(), 6);
    let mut words = output::initial(case.inputs.len(), case.words.len());
    for (thread, outcome) in expected.threads.iter().enumerate() {
        let Outcome::Words(result) = outcome else {
            panic!("thread {thread} unexpectedly panicked");
        };
        assert_eq!(
            &result[..4],
            &[0xffff_ffff, 0x7fff_ffff, 0x0000_0001, 0x8000_0000]
        );
        let start = thread * case.words.len();
        words[start..start + case.words.len()].copy_from_slice(result);
    }
    assert_eq!(case.gpu_mismatch(&expected, None, &words, &[]), None);
    for high in [1, 3] {
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
    let mut words = output::initial(1, 2);
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
