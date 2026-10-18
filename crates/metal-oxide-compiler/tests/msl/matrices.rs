use super::support;

const VALID: &str = "crates/metal-oxide-compiler/tests/fixtures/matrices.rs";
const INVALID: &str = "crates/metal-oxide-compiler/tests/fixtures/invalid_matrices.rs";
const UNIFORMITY: &str = "crates/metal-oxide-compiler/tests/fixtures/matrix_uniformity.rs";

#[test]
fn matrix_kernels_emit_simdgroup_matrices_and_record_their_capability() {
    let (output, directory) = support::emit(VALID, &["-C", "overflow-checks=off"]);
    support::checked(output);
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    for expected in [
        "simdgroup_matrix<float, 8, 8>",
        "simdgroup_matrix<half, 8, 8>",
        "simdgroup_matrix<bfloat, 8, 8>",
        "make_filled_simdgroup_matrix<float, 8, 8>(",
        "metal_oxide_matrix_multiply_accumulate(",
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

/// The MSL of kernel `name` with the numbers of its locals dropped, so `v12` reads `v`.
fn unnumbered_kernel(msl: &str, name: &str) -> String {
    let start = msl.find(&format!("kernel void {name}(")).unwrap();
    let kernel = &msl[start..];
    let end = kernel[1..]
        .find("kernel void ")
        .map_or(kernel.len(), |at| at + 1);
    let mut text = String::new();
    let mut previous = ' ';
    let mut characters = kernel[..end].chars().peekable();
    while let Some(character) = characters.next() {
        text.push(character);
        if character == 'v' && !previous.is_alphanumeric() && previous != '_' {
            while characters.next_if(char::is_ascii_digit).is_some() {}
        }
        previous = character;
    }
    text
}

#[test]
fn matrix_loads_and_stores_keep_their_flag_and_pass_column_before_row() {
    let (output, directory) = support::emit(VALID, &["-C", "overflow-checks=off"]);
    support::checked(output);
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    let kernel = |name| unnumbered_kernel(&msl, name);
    for (name, fragment) in [
        // `tile_b.load_matrix(0, 8, 16)`: row 0, column 8.
        (
            "gemm_f16",
            "metal_oxide_matrix_load<half>(v, ulong(16u), ulong2(8u, 0u), false);",
        ),
        // `tile_c.store_matrix(group * 8, 8, 16, acc[1])`: a row from a variable, column 8.
        (
            "gemm_f16",
            "simdgroup_store(v, v, ulong(16u), ulong2(8u, v), false);",
        ),
        // `tile_b.load_matrix_transposed(8, 0, 8)`: row 8, column 0.
        (
            "gemm_nt_f16",
            "metal_oxide_matrix_load<half>(v, ulong(8u), ulong2(0u, 8u), true);",
        ),
        (
            "gemm_nt_f16",
            "metal_oxide_matrix_load<half>(v, ulong(8u), ulong2(0u, 0u), true);",
        ),
    ] {
        let text = kernel(name);
        assert!(text.contains(fragment), "{name} lacks {fragment:?}");
    }
    // Of the four loads and seven stores, one load and two stores are transposed.
    let round_trip = kernel("round_trip_f32");
    let count = |operation: &str, flag: &str| {
        round_trip
            .matches(&format!("{operation}, {flag});"))
            .count()
    };
    let load = "metal_oxide_matrix_load<float>(v, ulong(8u), ulong2(0u, v)";
    let store = "simdgroup_store(v, v, ulong(8u), ulong2(0u, v)";
    assert_eq!((count(load, "false"), count(load, "true")), (3, 1));
    assert_eq!((count(store, "false"), count(store, "true")), (5, 2));
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
    rejects(
        "parameter",
        "unsupported kernel parameter type: metal_oxide_device::simdgroup::Matrix<f32>",
    );
    rejects(
        "element",
        "unsupported kernel parameter type: metal_oxide_device::ReadBuffer<",
    );
}

#[test]
fn group_uniform_control_and_checks_reach_matrix_operations() {
    let (output, directory) = support::emit(UNIFORMITY, &["-C", "overflow-checks=off"]);
    support::checked(output);
    assert!(directory.join("kernels.metal").exists());
    // The SIMD group index is not proved below `groups`, so `checked_group` keeps its check.
    let message = "SIMD group outside the launch contract";
    support::check(&directory, serde_json::json!({ "panic": message }), 15);
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
