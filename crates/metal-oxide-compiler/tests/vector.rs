#![cfg(feature = "rustc-private")]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[path = "vector/gpu.rs"]
mod gpu;
mod support;

#[test]
fn portable_vectors_lower_to_scalars() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/vector_basic.rs",
        &[],
    );
    support::checked(output);
    let source = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    for intrinsic in ["simd_add(", "simd_select(", "simd_reduce_"] {
        assert!(!source.contains(intrinsic));
    }
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert!(abi.required_features.is_empty());
}
