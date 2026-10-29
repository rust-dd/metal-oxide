use super::enclosing;

/// The inline modules around the first `ITEM` in `text`.
fn around(text: &str) -> Vec<String> {
    let (line, before) = text
        .lines()
        .enumerate()
        .find_map(|(index, line)| line.find("ITEM").map(|column| (index + 1, &line[..column])))
        .unwrap();
    enclosing(text, (line, before.chars().count() + 1))
}

#[test]
fn items_outside_inline_modules_have_no_module_path() {
    assert!(around("use a::b;\nITEM fn k() {}\n").is_empty());
    assert!(around("mod a {}\nmod b;\nfn f() { let x = 1; }\nITEM\n").is_empty());
    assert!(around("macro_rules! m { ($n:ident) => { mod $n {} }; }\nITEM\n").is_empty());
}

#[test]
fn inline_modules_name_the_path_outermost_first() {
    let nested = "mod a {\n    pub(crate) mod b {\n        ITEM\n    }\n}\n";
    assert_eq!(around(nested), ["a", "b"]);
    assert_eq!(
        around("#[cfg(test)]\nmod tests { mod r#type { ITEM } }\n"),
        ["tests", "type"]
    );
    assert_eq!(around("mod a { fn f() {} }\nmod b { ITEM }\n"), ["b"]);
    assert_eq!(around("fn f() {\n    mod inner { ITEM }\n}\n"), ["inner"]);
    assert_eq!(around("mod é { ITEM }\n"), ["é"]);
}

#[test]
fn braces_in_comments_and_literals_open_no_scope() {
    let text = r##"
mod a {
    // mod x {
    /* mod y { /* nested } */ { */
    const S: &str = "mod z { \" {";
    const R: &str = r#"} mod w {"#;
    const B: &[u8] = b"{" ;
    const C: [char; 3] = ['{', '\'', '\u{7d}'];
    const D: u8 = b'}';
    fn f<'a>(x: &'a str) -> &'a str { 'outer: loop { break 'outer x; } }
    ITEM
}
"##;
    assert_eq!(around(text), ["a"]);
}
