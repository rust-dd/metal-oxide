use super::notes;

#[test]
fn notes_keep_the_locations_they_name() {
    let details = [
        "  --> src/main.rs:8:21",
        "   |",
        "8  |     let total = values.into_iter().sum::<u32>();",
        "   |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        "   |",
        "note: in code from another crate",
        "  --> /rust/library/core/src/iter/traits/accum.rs:52:17",
        "   |",
        "   = note: this error originates in the macro `integer_sum_product`",
    ];
    assert_eq!(
        notes(&details),
        "\nnote: in code from another crate (/rust/library/core/src/iter/traits/accum.rs:52:17)\
         \nnote: this error originates in the macro `integer_sum_product`"
    );
}
