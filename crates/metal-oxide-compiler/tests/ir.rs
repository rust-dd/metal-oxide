mod support;

#[test]
fn invalid_output_flag_returns_failure() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_metal-oxide-compiler"))
        .arg("--metal-output")
        .output()
        .unwrap();
    support::rejected(output, "requires a directory");
}

#[test]
fn vec_add_imports_device_access_and_preserves_checks() {
    let (output, directory) = support::emit("examples/vec-add/kernels/src/lib.rs", &[]);
    support::rejected(output, "MIR assertion");
    let ir = std::fs::read_to_string(directory.join("kernels.oxide-ir")).unwrap();
    for operation in [
        "BufferLoad",
        "BufferStore",
        "BlockIdx",
        "BlockDim",
        "ThreadIdx",
        "Assert",
        "enabled: true",
        "Checked(",
        "Read",
        "Write",
        "Device",
    ] {
        assert!(ir.contains(operation), "missing {operation}:\n{ir}");
    }
}

#[test]
fn generic_helpers_import_concrete_types_and_assert_policy() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/generics.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let ir = std::fs::read_to_string(directory.join("kernels.oxide-ir")).unwrap();
    for value in [
        "sum::<f32>",
        "sum::<u32>",
        "bias::<3>",
        "enabled: false",
        "GridDim",
    ] {
        assert!(ir.contains(value), "missing {value}:\n{ir}");
    }
}

#[test]
fn nested_branches_early_return_and_loop_import_from_real_mir() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/m2_control_flow.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let ir = std::fs::read_to_string(directory.join("kernels.oxide-ir")).unwrap();
    assert!(ir.contains("compute"));
    assert!(ir.contains("Branch"));
    assert!(ir.contains("Return"));
    assert!(ir.contains("Call"));
}

#[test]
fn unsupported_local_type_has_a_rust_source_diagnostic() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/m2_f64.rs",
        &["-C", "overflow-checks=off"],
    );
    support::rejected(output, "unsupported device type: f64");
    assert!(!directory.join("kernels.metal").exists());
}

#[test]
fn division_checks_stay_enabled_when_overflow_checks_are_off() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/m2_div.rs",
        &["-C", "overflow-checks=off"],
    );
    support::rejected(output, "MIR assertion");
    let ir = std::fs::read_to_string(directory.join("kernels.oxide-ir")).unwrap();
    assert!(ir.contains("Assert"));
    assert!(ir.contains("enabled: true"));
}
