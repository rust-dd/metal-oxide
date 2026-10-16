#![cfg(feature = "rustc-private")]

mod support;

fn rejects(fixture: &str, case: &str, message: &str) {
    let source = format!("crates/metal-oxide-compiler/tests/fixtures/rejections/{fixture}.rs");
    let options = if case.is_empty() {
        vec![]
    } else {
        vec!["--cfg", case]
    };
    let (output, directory) = support::emit(&source, &options);
    support::rejected(output, message);
    assert!(
        !directory.join("kernels.metal").exists(),
        "{fixture}/{case}"
    );
}

#[test]
fn vectors_reject_unsupported_elements_and_lane_counts() {
    for case in ["single", "wide", "float64", "signed64", "unsigned64"] {
        rejects(
            "vector_types",
            case,
            "vectors require 2-4 lanes of f32 or 8/16/32-bit integers",
        );
    }
}

#[test]
fn vectors_cannot_cross_the_abi_directly_or_inside_aggregates() {
    for case in ["direct", "mask", "buffer", "nested", "nested_buffer"] {
        rejects("vector_abi", case, "unsupported kernel parameter type");
    }
}

#[test]
fn ordering_cannot_cross_the_abi_directly_or_in_a_buffer() {
    for case in ["ordering", "ordering_buffer"] {
        rejects("vector_abi", case, "unsupported kernel parameter type");
    }
}

#[test]
fn empty_arrays_reject_indexing_without_storage() {
    rejects(
        "empty_array_index",
        "",
        "array index requires array storage",
    );
}

#[test]
fn vector_memory_operations_are_rejected() {
    for case in ["gather", "scatter", "masked_load", "masked_store"] {
        rejects(
            "vector_memory",
            case,
            "vectors require 2-4 lanes of f32 or 8/16/32-bit integers",
        );
    }
    for (case, message) in [
        (
            "masked_load_ptr",
            "invalid SIMD intrinsic signature: simd_masked_load",
        ),
        (
            "masked_store_ptr",
            "SIMD intrinsics require repr(simd) vectors",
        ),
    ] {
        rejects("vector_memory", case, message);
    }
}

#[test]
fn unsupported_vector_intrinsics_are_diagnosed() {
    rejects(
        "vector_intrinsic",
        "",
        "unsupported vector intrinsic: simd_fsqrt",
    );
}

#[test]
fn invalid_vector_intrinsic_arguments_are_diagnosed() {
    for case in ["nonvector", "representation", "array"] {
        rejects(
            "vector_intrinsic_arguments",
            case,
            "SIMD intrinsics require repr(simd) vectors",
        );
    }
    for (case, message) in [
        ("lane_mismatch", "invalid SIMD intrinsic signature: simd_eq"),
        (
            "destination",
            "invalid SIMD intrinsic signature: simd_shuffle",
        ),
        ("shuffle_bounds", "vector shuffle index is out of bounds"),
        (
            "shuffle_dynamic",
            "3rd argument of `simd_shuffle` is required to be a `const` item",
        ),
        (
            "shuffle_signed",
            "invalid SIMD intrinsic signature: simd_shuffle",
        ),
    ] {
        rejects("vector_intrinsic_arguments", case, message);
    }
}

#[test]
fn vector_bitmasks_require_integer_lanes_and_one_byte() {
    for case in [
        "bitmask_array",
        "bitmask_float",
        "bitmask_bool",
        "bitmask_wide",
        "bitmask_float_lanes",
    ] {
        rejects(
            "vector_intrinsic_arguments",
            case,
            "vector bitmask requires integer lanes and a u8 result",
        );
    }
}

#[test]
fn vector_intrinsics_validate_mask_and_scalar_types() {
    for (case, intrinsic) in [
        ("comparison_float", "simd_eq"),
        ("comparison_width", "simd_eq"),
        ("select_float", "simd_select"),
        ("select_lanes", "simd_select"),
        ("reduction_output", "simd_reduce_add_unordered"),
        ("reduction_initial", "simd_reduce_add_ordered"),
        ("reduction_boolean", "simd_reduce_all"),
        ("extract_output", "simd_extract"),
        ("insert_value", "simd_insert"),
        ("cast_lanes", "simd_as"),
        ("splat_type", "simd_splat"),
    ] {
        rejects(
            "vector_intrinsic_arguments",
            case,
            &format!("invalid SIMD intrinsic signature: {intrinsic}"),
        );
    }
}
