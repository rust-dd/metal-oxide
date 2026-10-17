use super::support;

const VALID: &str = "crates/metal-oxide-compiler/tests/fixtures/matrices.rs";
const INVALID: &str = "crates/metal-oxide-compiler/tests/fixtures/invalid_matrices.rs";

#[test]
fn matrix_kernels_emit_simdgroup_matrices_and_record_their_capability() {
    let (output, directory) = support::emit(VALID, &["-C", "overflow-checks=off"]);
    support::checked(output);
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    for expected in [
        "simdgroup_matrix<float, 8, 8>",
        "simdgroup_matrix<half, 8, 8>",
        "simdgroup_matrix<bfloat, 8, 8>",
        "metal_oxide_matrix_load<half>(",
        "make_filled_simdgroup_matrix<float, 8, 8>(",
        "metal_oxide_matrix_multiply_accumulate(",
        "simdgroup_store(",
    ] {
        assert!(msl.contains(expected), "missing {expected:?}");
    }
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        abi.required_features,
        [
            "simd_groups",
            "simdgroup_matrix",
            "dynamic_threadgroup_memory",
            "bfloat16"
        ]
    );
}

/// Emits `INVALID` with `cfg` set and checks the compiler rejects it with `message`.
pub(super) fn rejects(cfg: &str, message: &str) {
    let (output, directory) = support::emit(INVALID, &["-C", "overflow-checks=off", "--cfg", cfg]);
    support::rejected(output, message);
    assert!(!directory.join("kernels.metal").exists(), "{cfg}");
}

#[test]
fn matrix_kernels_reject_divergence_and_abi_crossings() {
    rejects("lane_branch", "uniform participation");
    rejects("parameter", "unsupported kernel parameter type");
    rejects("element", "unsupported kernel parameter type");
}

#[test]
fn group_uniform_control_and_checks_reach_matrix_operations() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/matrix_uniformity.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    assert!(directory.join("kernels.metal").exists());
}

#[test]
fn matrix_operands_and_blocks_follow_simd_group_rules() {
    for cfg in ["lane_operand", "lane_value", "lane_matrix", "by_reference"] {
        rejects(cfg, "uniform within the SIMD group");
    }
    for cfg in ["no_block", "odd_block"] {
        rejects(cfg, "multiple of 32");
    }
}
