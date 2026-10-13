use super::support;
use metal_oxide_artifact::{Abi, MathMode};

const FIXTURE: &str = "crates/metal-oxide-compiler/tests/fixtures/transcendentals.rs";

#[test]
fn numerical_modes_change_msl_and_binding_identity() {
    let mut artifacts = Vec::new();
    for mode in [None, Some("precise"), Some("fast")] {
        let mut arguments = vec!["-C", "overflow-checks=off"];
        if let Some(mode) = mode {
            arguments.extend(["--metal-math-mode", mode]);
        }
        let (output, directory) = support::emit(FIXTURE, &arguments);
        support::checked(output);
        let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
        let abi =
            Abi::from_json(&std::fs::read_to_string(directory.join("abi.json")).unwrap()).unwrap();
        let bindings = std::fs::read_to_string(directory.join("bindings.rs")).unwrap();
        let expected = if mode == Some("fast") {
            MathMode::Fast
        } else {
            MathMode::Precise
        };
        assert_eq!(abi.math_mode, expected);
        for name in ["sqrt", "log", "exp", "sin", "cos", "pow", "tanh", "atan"] {
            assert!(
                msl.contains(&format!("{}::{name}(", expected.as_str())),
                "{name}"
            );
        }
        assert_eq!(
            msl.contains("math_mode(relaxed)"),
            expected == MathMode::Fast
        );
        assert_eq!(msl.contains("contract(fast)"), expected == MathMode::Fast);
        assert_eq!(
            msl.contains("FP_CONTRACT OFF"),
            expected == MathMode::Precise
        );
        artifacts.push((msl, abi, bindings));
    }
    assert_eq!(artifacts[0], artifacts[1]);
    assert_ne!(artifacts[0].2, artifacts[2].2);
}

#[test]
fn invalid_or_repeated_numerical_modes_fail_before_emission() {
    for arguments in [
        vec!["--metal-math-mode"],
        vec!["--metal-math-mode", "relaxed"],
        vec!["--metal-math-mode", "fast", "--metal-math-mode", "precise"],
    ] {
        let (output, directory) = support::emit(FIXTURE, &arguments);
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("math"), "{error}");
        assert!(!error.contains("internal compiler error"), "{error}");
        assert!(!directory.join("kernels.metal").exists());
    }
}

#[test]
fn fast_math_keeps_integer_runtime_checks() {
    for (source, overflow, kind) in [
        (
            "crates/metal-oxide-compiler/tests/fixtures/runtime_checks.rs",
            "on",
            serde_json::json!({"overflow": "add"}),
        ),
        (
            "crates/metal-oxide-compiler/tests/fixtures/m2_div.rs",
            "off",
            serde_json::json!("division_by_zero"),
        ),
    ] {
        let setting = format!("overflow-checks={overflow}");
        let (output, directory) =
            support::emit(source, &["--metal-math-mode", "fast", "-C", &setting]);
        support::checked(output);
        let checks = std::fs::read_to_string(directory.join("checks.json")).unwrap();
        let checks = serde_json::from_str::<Vec<serde_json::Value>>(&checks).unwrap();
        assert!(
            checks.iter().any(|check| check["kind"] == kind),
            "{checks:?}"
        );
        let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
        assert!(msl.contains("metal_oxide_fail(metal_oxide_ctx"), "{msl}");
    }
}
