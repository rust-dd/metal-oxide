use super::support;

#[test]
fn read_write_buffer_runs_in_place_and_has_exclusive_bindings() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/read_write.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = std::fs::read_to_string(directory.join("abi.json")).unwrap();
    assert!(abi.contains("read_write"), "{abi}");
    let bindings = std::fs::read_to_string(directory.join("bindings.rs")).unwrap();
    assert!(bindings.contains("&mut metal_oxide::Buffer<f32>"));
    assert!(bindings.contains("Argument::read_write"));
    assert_eq!(
        support::execute_msl(
            &directory,
            r#"
        float values[6] = {3, 1, -2, 4, 99, 99};
        uint pairs = 2;
        for (uint i = 0; i < 3; ++i)
            fft_stage(values, pairs, {i,0,0}, {0,0,0}, {3,1,1}, {1,1,1});
        float want[6] = {4, 2, 2, -6, 99, 99};
        for (uint i = 0; i < 6; ++i) if (values[i] != want[i]) return 1;
        std::cout << "ok";
    "#
        ),
        "ok"
    );
}
