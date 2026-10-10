use super::support;

#[test]
fn unknown_u64_overflow_is_rejected() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/u64_checked.rs",
        &[],
    );
    support::rejected(output, "MIR assertion");
    assert!(!directory.join("kernels.metal").exists());
}

#[test]
fn u64_arithmetic_keeps_high_bits_and_reports_overflow() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/u64.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    assert_eq!(
        support::execute_msl(
            &directory,
            r#"
        ulong out[15] = {};
        ulong a = 0xfffffffffffffffful, b = 2;
        uint shift = 65; int signed_value = -1;
        wide_ops(out,a,b,shift,signed_value,{0,0,0},{0,0,0},{1,1,1},{1,1,1});
        ulong expected[15] = {1, a-1, a-2, 0, a>>1, a, 0xfffffffful, 1, 1, 0, a/2, 1, a, a, 7};
        for (uint i = 0; i < 15; ++i) if (out[i] != expected[i]) return 1;
        std::cout << "ok";
    "#
        ),
        "ok"
    );
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    let kernel = abi
        .kernels
        .iter()
        .find(|kernel| kernel.name == "splitmix")
        .unwrap();
    let metal_oxide_artifact::ParameterType::Value { layout } = &kernel.parameters[1].ty else {
        panic!()
    };
    assert_eq!((layout.size, layout.alignment), (8, 8));
}
