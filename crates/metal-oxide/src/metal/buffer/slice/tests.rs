use super::elements;

#[test]
fn ranges_resolve_to_a_start_and_a_count() {
    assert_eq!(elements(&.., 5), (0, 5));
    assert_eq!(elements(&(2..), 5), (2, 3));
    assert_eq!(elements(&(..3), 5), (0, 3));
    assert_eq!(elements(&(1..4), 5), (1, 3));
    assert_eq!(elements(&(1..=3), 5), (1, 3));
    assert_eq!(elements(&(5..), 5), (5, 0));
}

#[test]
#[should_panic(expected = "range end index 6 out of range for a buffer of length 5")]
fn ranges_past_the_end_panic() {
    elements(&(..6), 5);
}

#[test]
#[should_panic(expected = "slice index starts at 3 but ends at 2")]
fn decreasing_ranges_panic() {
    let (start, end) = (3, 2);
    elements(&(start..end), 5);
}
