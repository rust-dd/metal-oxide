use super::*;

#[test]
fn summary_preserves_spread_and_rejects_empty_samples() {
    assert!(Summary::new(&[]).is_err());
    let summary = Summary::new(&[100, 3, 1, 4, 2]).unwrap();
    assert_eq!((summary.min, summary.median, summary.max), (1, 3, 100));
    assert_eq!((summary.p10, summary.p90), (1, 100));
    assert_eq!(summary.mean, 22.0);
    let summary = Summary::new(&[u64::MAX - 2, u64::MAX]).unwrap();
    assert_eq!(summary.median, u64::MAX - 1);
}
