/// Misused kernel macros fail before any device build, with diagnostics on the tokens at fault.
#[test]
fn kernel_macros_report_misuse_on_its_tokens() {
    trybuild::TestCases::new().compile_fail("tests/ui/*.rs");
}
