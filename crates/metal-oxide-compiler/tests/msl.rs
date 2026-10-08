mod support;

#[path = "msl/collectives.rs"]
mod collectives;
#[path = "msl/control_flow.rs"]
mod control_flow;

#[test]
fn atomic_operations_compile_with_device_and_threadgroup_memory() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/atomic_ops.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.kernels.len(), 5);
    assert_eq!(abi.required_features, ["int32_atomics"]);
    assert_eq!(abi.kernels[0].parameters[0].binding, 0);
    assert_eq!(
        abi.kernels
            .iter()
            .find(|k| k.name == "shared_counter")
            .unwrap()
            .required_block,
        Some([256, 1, 1])
    );
}

#[test]
fn records_with_references_are_rejected() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/record_pointer.rs",
        &[],
    );
    support::rejected(output, "unsupported device type");
    assert!(!directory.join("kernels.metal").exists());
}

#[test]
fn scalar_record_helpers_execute_from_generated_msl() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/records.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let result = support::execute_msl(
        &directory,
        "float a[1] = {2}; float b[1] = {5}; float sum[1] = {}; float product[1] = {}; uint n=1; record_math(a,b,sum,product,n,{0,0,0},{0,0,0},{1,1,1},{1,1,1}); std::cout << sum[0] << ',' << product[0];",
    );
    assert_eq!(result, "21,31");
}

#[test]
fn narrow_overflow_checks_are_preserved() {
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/narrow.rs", &[]);
    support::rejected(output, "MIR assertion");
    assert!(!directory.join("kernels.metal").exists());
}

#[test]
fn threadgroup_kernels_emit_shape_and_typed_bindings() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/cooperative.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        abi.kernels
            .iter()
            .find(|k| k.name == "reduce")
            .unwrap()
            .required_block,
        Some([256, 1, 1])
    );
    let bindings = std::fs::read_to_string(directory.join("bindings.rs")).unwrap();
    assert!(bindings.contains("LaunchConfig<256, 1, 1>"));
    assert!(bindings.contains("Argument::atomic"));
}

#[test]
fn divergent_threadgroup_participation_is_rejected() {
    for fixture in [
        "divergent_barrier",
        "early_barrier",
        "loop_barrier",
        "divergent_simd",
    ] {
        let (output, directory) = support::emit(
            &format!("crates/metal-oxide-compiler/tests/fixtures/{fixture}.rs"),
            &["-C", "overflow-checks=off"],
        );
        support::rejected(output, "uniform participation");
        assert!(!directory.join("kernels.metal").exists());
    }
}

#[test]
fn simd_requirements_are_recorded_in_the_abi() {
    let (output, directory) = support::emit(
        "examples/matmul/kernels/src/lib.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.required_features, ["simd_groups"]);
}

#[test]
fn parameter_names_produce_valid_metadata_and_rust_bindings() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/m3_names.rs",
        &[],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        abi.kernels[0]
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["arg_0", "type", "arg_2"]
    );
    let bindings = std::fs::read_to_string(directory.join("bindings.rs")).unwrap();
    assert!(bindings.contains("r#type: f32"));
}

#[test]
fn rust_vec_add_generated_source_executes_and_guards_padding() {
    let (output, directory) = support::emit(
        "examples/vec-add/kernels/src/lib.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let output = support::execute_msl(
        &directory,
        r#"
        float a[8] = {1,2,3,4,5,6,7,8};
        float b[8] = {10,20,30,40,50,60,70,80};
        float out[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
        uint n = 3;
        for (uint i = 0; i < 8; ++i) vec_add(a,b,out,n,{i%4,0,0},{i/4,0,0},{4,1,1},{2,1,1});
        for (float value : out) std::cout << value << ',';
    "#,
    );
    assert_eq!(output, "11,22,33,-1,-1,-1,-1,-1,");
}

#[test]
fn real_mir_nested_branches_loop_and_early_return_execute() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/m2_control_flow.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let output = support::execute_msl(
        &directory,
        r#"
        uint out[12] = {500,500,500,500,500,500,500,500,500,500,500,500};
        uint n = 8;
        for (uint i = 0; i < 12; ++i) control_flow(out,n,{i%4,0,0},{i/4,0,0},{4,1,1},{3,1,1});
        for (uint value : out) std::cout << value << ',';
    "#,
    );
    assert_eq!(output, "99,3,10,15,22,25,32,32,500,500,500,500,");
}

#[test]
fn concrete_trait_and_const_generic_helpers_execute() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/generics.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let output = support::execute_msl(
        &directory,
        r#"
        float f[2] = {-1,-1}; uint i[1] = {999};
        helpers(f,i,{0,0,0},{0,0,0},{1,1,1},{1,1,1});
        std::cout << f[0] << ',' << f[1] << ',' << i[0];
    "#,
    );
    assert_eq!(output, "-1,3,6");
}

#[test]
fn checked_rust_build_does_not_emit_an_unchecked_shader() {
    let (output, directory) = support::emit("examples/vec-add/kernels/src/lib.rs", &[]);
    support::rejected(output, "MIR assertion");
    assert!(!directory.join("kernels.metal").exists());
}

#[test]
fn integer_division_assert_is_rejected_in_wrapping_builds() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/m2_div.rs",
        &["-C", "overflow-checks=off"],
    );
    support::rejected(output, "MIR assertion");
    assert!(!directory.join("kernels.metal").exists());
}

#[test]
fn unit_copy_preserves_helper_side_effects() {
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/m2_unit.rs", &[]);
    support::checked(output);
    let output = support::execute_msl(
        &directory,
        r#"
        uint out[1] = {0};
        unit_copy(out,{0,0,0},{0,0,0},{1,1,1},{1,1,1});
        std::cout << out[0];
    "#,
    );
    assert_eq!(output, "7");
}

#[test]
fn rust_raw_identifier_cannot_be_an_msl_keyword() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/m2_reserved.rs",
        &[],
    );
    support::rejected(output, "not a supported MSL identifier");
    assert!(!directory.join("kernels.metal").exists());
}

#[test]
fn failed_rebuild_removes_previous_generated_source() {
    let (output, directory) = support::emit(
        "examples/vec-add/kernels/src/lib.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    assert!(directory.join("kernels.metal").exists());
    let output = support::emit_into(
        "crates/metal-oxide-compiler/tests/fixtures/m2_f64.rs",
        &[],
        &directory,
    );
    support::rejected(output, "unsupported device type: f64");
    assert!(!directory.join("kernels.metal").exists());
    assert!(!directory.join("kernels.oxide-ir").exists());
    assert!(!directory.join("abi.json").exists());
    assert!(!directory.join("bindings.rs").exists());
}

#[test]
fn informational_commands_preserve_generated_files() {
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/m2_unit.rs", &[]);
    support::checked(output);
    let msl = std::fs::read(directory.join("kernels.metal")).unwrap();
    let ir = std::fs::read(directory.join("kernels.oxide-ir")).unwrap();
    let response = directory.join("help.args");
    std::fs::write(&response, "--help\ntests/fixtures/m2_unit.rs\n").unwrap();
    let response_argument = format!("@{}", response.display());
    let inspection = directory.join("inspect.rs");
    std::fs::write(&inspection, "fn main() {}\n").unwrap();
    let inspection = inspection.to_str().unwrap();
    for arguments in [
        vec![],
        vec!["--help"],
        vec!["--help", "tests/fixtures/m2_unit.rs"],
        vec!["-vh", "tests/fixtures/m2_unit.rs"],
        vec![response_argument.as_str()],
        vec!["-C", "help", "tests/fixtures/m2_unit.rs"],
        vec!["-Chelp", "tests/fixtures/m2_unit.rs"],
        vec!["-Z", "help", "tests/fixtures/m2_unit.rs"],
        vec!["-Zhelp", "tests/fixtures/m2_unit.rs"],
        vec!["--version"],
        vec!["--print=sysroot"],
        vec!["--print", "sysroot"],
        vec!["-W", "help"],
        vec!["-Zunpretty=expanded", inspection],
        vec!["-Zunpretty=mir", inspection],
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_metal-oxide-compiler"))
            .arg("--metal-output")
            .arg(&directory)
            .args(arguments)
            .output()
            .unwrap();
        support::checked(output);
        assert_eq!(std::fs::read(directory.join("kernels.metal")).unwrap(), msl);
        assert_eq!(
            std::fs::read(directory.join("kernels.oxide-ir")).unwrap(),
            ir
        );
    }
}

#[test]
fn atomic_threadgroup_storage_rejects_float_elements() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/invalid_atomic_shared.rs",
        &[],
    );
    support::rejected(output, "unsupported local type");
    assert!(!directory.join("kernels.metal").exists());
}
