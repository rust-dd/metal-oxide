mod support;

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
}

#[test]
fn informational_commands_preserve_generated_files() {
    let (output, directory) =
        support::emit("crates/metal-oxide-compiler/tests/fixtures/m2_unit.rs", &[]);
    support::checked(output);
    let msl = std::fs::read(directory.join("kernels.metal")).unwrap();
    let ir = std::fs::read(directory.join("kernels.oxide-ir")).unwrap();
    for arguments in [
        vec!["--help"],
        vec!["--version"],
        vec!["--print=sysroot"],
        vec!["--print", "sysroot"],
        vec!["-W", "help"],
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
