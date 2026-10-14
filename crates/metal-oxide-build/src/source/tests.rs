use super::blank;

#[test]
fn blanking_keeps_the_kept_range_at_its_lines_and_columns() {
    let text = "use a::b;\n#[kernel] fn f(x: u32) {\n    é(x)\n}\nfn main() {}\n";
    assert_eq!(
        blank(text, ((2, 11), (4, 2)), None),
        "\n          fn f(x: u32) {\n    é(x)\n}\n"
    );
    let edited = "use a::c;\n#[kernel] fn f(x: u32) {\n    é(x)\n}\nfn other() {}\n";
    assert_eq!(
        blank(edited, ((2, 11), (4, 2)), None),
        blank(text, ((2, 11), (4, 2)), None)
    );
    let module = "#[a] #[kernels] mod m {}\n";
    assert_eq!(
        blank(module, ((1, 1), (1, 25)), Some(((1, 6), (1, 16)))),
        "#[a]            mod m {}\n"
    );
}
