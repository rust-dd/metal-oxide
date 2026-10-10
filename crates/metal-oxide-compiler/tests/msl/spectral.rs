use super::support;

#[test]
fn guarded_stack_bounds_survive_loop_convergence() {
    let (output, _) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/guarded_stack.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
}

#[test]
fn spectral_tiles_and_fused_radix_stages_compile_from_rust() {
    let (output, directory) = support::emit(
        "examples/spectral/kernels/src/lib.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    for entry in [
        "gen_tile_fft",
        "fft_radix4",
        "fft_radix4_extract",
        "fft_butterfly",
        "fft_butterfly_extract",
        "sample_jumps",
        "history_program",
        "series_clock",
    ] {
        assert!(abi.kernels.iter().any(|kernel| kernel.name == entry));
    }
}
