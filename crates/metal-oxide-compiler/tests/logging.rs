#![cfg(feature = "rustc-private")]

mod support;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/logging.rs";

#[test]
fn log_messages_lower_to_os_log_with_typed_specifiers() {
    let (output, directory) = support::emit(FIXTURE, &[]);
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.required_features, ["logging"]);
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    assert!(msl.contains("#include <metal_logging>"), "{msl}");
    assert!(
        msl.contains(
            r#"os_log_default.log("thread %u scale %.9g half %.5g wide %lu signed %d %ld {braces} 100%%""#
        ),
        "{msl}"
    );
    let output = support::execute_msl(
        &directory,
        "uint out[2] = {}; for (uint i : {0u, 1u}) logging(out, 0.1f, half(0.5f), {i,0,0}, {}, {2,1,1}, {1,1,1}, metal_oxide_gate);",
    );
    assert_eq!(
        output,
        "thread 0 scale 0.100000001 half 0.5 wide 1099511627776 signed -3 -1099511627776 {braces} 100%\n\
         thread 1 scale 0.100000001 half 0.5 wide 1099511627776 signed -3 -1099511627776 {braces} 100%\n"
    );
}

#[test]
fn log_formats_need_matching_plain_placeholders() {
    // The error names the `log!` call, not the device crate's macro that expands it.
    for (configuration, message, line) in [
        ("mismatch", "log format takes 2 arguments, got 1", 15),
        ("debug_format", "only `{}` placeholders", 21),
    ] {
        let (output, _) = support::emit(FIXTURE, &["--cfg", configuration]);
        let error = String::from_utf8_lossy(&output.stderr).into_owned();
        support::rejected(output, message);
        assert!(
            error.contains(&format!("fixtures/logging.rs:{line}:5")),
            "{error}"
        );
    }
}
