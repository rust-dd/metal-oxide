use std::{cell::RefCell, rc::Rc};

use super::*;

type Log = Rc<RefCell<Vec<String>>>;

fn config(warmup: usize, samples: usize) -> Config {
    serde_json::from_value(serde_json::json!({
        "artifacts": {}, "references": {}, "cases": ["sort"], "samples": samples,
        "warmup": warmup, "elements": 1, "matrix": [1, 1, 1], "gemm": [1, 1, 1],
        "gemm_repeat": 1, "sort": 1,
    }))
    .unwrap()
}

/// Logs `<variant> warmup|timed` per sample, fails the `fail_on`th (from 0) and keeps each timed
/// sample as the number of samples taken so far.
struct Fake {
    case: Case,
    log: Log,
    taken: usize,
    fail_on: Option<usize>,
}

impl Fake {
    fn new(variant: &str, log: &Log) -> Self {
        Self {
            case: Case::new("sort", variant, vec![1]),
            log: Rc::clone(log),
            taken: 0,
            fail_on: None,
        }
    }
}

impl Candidate for Fake {
    fn sample(&mut self, keep: bool) -> Result<()> {
        let phase = if keep { "timed" } else { "warmup" };
        let variant = self.case.variant.clone();
        self.log.borrow_mut().push(format!("{variant} {phase}"));
        self.taken += 1;
        if self.fail_on == Some(self.taken - 1) {
            return Err(format!("{variant} failed").into());
        }
        if keep {
            self.case.samples.push(Sample {
                submit_ns: self.taken as u64,
                ..Sample::default()
            });
        }
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<Case> {
        self.case.finish()
    }
}

fn fakes(variants: &[&str], log: &Log) -> Vec<Box<dyn Candidate>> {
    variants
        .iter()
        .map(|variant| Box::new(Fake::new(variant, log)) as Box<dyn Candidate>)
        .collect()
}

fn submits(case: &Case) -> Vec<u64> {
    case.samples.iter().map(|sample| sample.submit_ns).collect()
}

#[test]
fn every_round_takes_each_candidate_once_starting_one_further() {
    let order = |count, round| round_order(count, round).collect::<Vec<_>>();
    assert_eq!(order(3, 0), [0, 1, 2]);
    assert_eq!(order(3, 1), [1, 2, 0]);
    assert_eq!(order(3, 2), [2, 0, 1]);
    assert_eq!(order(3, 3), [0, 1, 2]);
    assert_eq!(order(2, 5), [1, 0]);
    assert_eq!(order(1, 7), [0]);
    assert!(order(0, 4).is_empty());
}

#[test]
fn interleaving_rotates_the_candidates_through_warmup_and_timed_rounds() {
    let log = Log::default();
    let cases = interleave(&config(1, 2), fakes(&["rust", "msl", "mps"], &log)).unwrap();
    let expected = [
        ["rust warmup", "msl warmup", "mps warmup"],
        ["msl timed", "mps timed", "rust timed"],
        ["mps timed", "rust timed", "msl timed"],
    ];
    assert_eq!(log.borrow()[..], expected.concat());
    let variants = cases.iter().map(|case| case.variant.as_str());
    assert_eq!(variants.collect::<Vec<_>>(), ["rust", "msl", "mps"]);
    for case in &cases {
        assert_eq!(submits(case), [2, 3], "{}", case.variant);
    }
}

#[test]
fn without_warmup_every_round_is_timed() {
    let log = Log::default();
    let cases = interleave(&config(0, 2), fakes(&["rust", "msl"], &log)).unwrap();
    let expected = ["rust timed", "msl timed", "msl timed", "rust timed"];
    assert_eq!(log.borrow()[..], expected);
    assert!(cases.iter().all(|case| submits(case) == [1, 2]));
}

#[test]
fn a_failing_sample_ends_the_run_with_its_error() {
    let log = Log::default();
    let mut candidates = fakes(&["rust"], &log);
    let msl = Fake {
        fail_on: Some(1),
        ..Fake::new("msl", &log)
    };
    candidates.push(Box::new(msl));
    let error = interleave(&config(1, 3), candidates).err().unwrap();
    assert_eq!(error.to_string(), "msl failed");
    let expected = ["rust warmup", "msl warmup", "msl timed"];
    assert_eq!(log.borrow()[..], expected);
}

#[test]
fn rust_and_msl_builds_both_candidates_before_the_first_sample() {
    let log = Log::default();
    let build = |variant: &str| {
        log.borrow_mut().push(format!("build {variant}"));
        Ok(Fake::new(variant, &log))
    };
    let cases = rust_and_msl(&config(1, 1), build).unwrap();
    let expected = [
        "build rust",
        "build msl",
        "rust warmup",
        "msl warmup",
        "msl timed",
        "rust timed",
    ];
    assert_eq!(log.borrow()[..], expected);
    let variants = cases.iter().map(|case| case.variant.as_str());
    assert_eq!(variants.collect::<Vec<_>>(), ["rust", "msl"]);
}

#[test]
fn rust_and_msl_runs_nothing_when_a_variant_fails_to_build() {
    let log = Log::default();
    let build = |variant: &str| {
        log.borrow_mut().push(format!("build {variant}"));
        if variant == "msl" {
            return Err("no metallib".into());
        }
        Ok(Fake::new(variant, &log))
    };
    let error = rust_and_msl(&config(1, 1), build).err().unwrap();
    assert_eq!(error.to_string(), "no metallib");
    assert_eq!(log.borrow()[..], ["build rust", "build msl"]);
}
