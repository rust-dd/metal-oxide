#![cfg(feature = "rustc-private")]

mod support;

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/diagnostics/core_rejections.rs";

/// The location of the first diagnostic line after `header` that starts with `-->`.
fn location_after<'a>(error: &'a str, header: &str) -> &'a str {
    error
        .lines()
        .skip_while(|line| !line.starts_with(header))
        .find_map(|line| line.trim_start().strip_prefix("--> "))
        .unwrap_or_else(|| panic!("no location after {header}:\n{error}"))
}

#[test]
fn rejections_in_core_report_at_the_kernel_call_that_reached_them() {
    // Each stage that rejects code: the importer, the collector before it and the prover after.
    for (configuration, message, line, column) in [
        ("import", "array iterators support only `next`", 8, 37),
        ("collector", "unsupported function instance", 15, 5),
        (
            "proof",
            "slice index is not below a checked slice length",
            24,
            38,
        ),
    ] {
        let (output, _) = support::emit(FIXTURE, &["--cfg", configuration]);
        let error = String::from_utf8_lossy(&output.stderr).into_owned();
        support::rejected(output, message);
        let call = location_after(&error, &format!("error: {message}"));
        assert!(
            call.ends_with(&format!("core_rejections.rs:{line}:{column}")),
            "{configuration}: {call}\n{error}"
        );
        let reached = location_after(&error, "note: in code from another crate");
        assert!(
            reached.contains("/library/core/src/"),
            "{configuration}: {reached}\n{error}"
        );
    }
}
