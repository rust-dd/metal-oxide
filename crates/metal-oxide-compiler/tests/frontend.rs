mod support;

#[test]
fn device_target_has_explicit_scalar_and_pointer_layout() {
    let report = support::checked(support::compile(
        "crates/metal-oxide-compiler/tests/fixtures/empty.rs",
        "metadata",
        false,
    ));
    assert!(
        report.contains("pointer=64 usize=64 endian=little"),
        "{report}"
    );
    for ty in ["f32", "u32", "i32"] {
        assert!(
            report.contains(&format!("{ty}: size=4 align=4")),
            "{report}"
        );
    }
}

#[test]
fn native_output_is_rejected() {
    support::rejected(
        support::compile(
            "crates/metal-oxide-compiler/tests/fixtures/empty.rs",
            "obj",
            false,
        ),
        "native code generation is not supported; use --emit=metadata with --metal-output",
    );
}

#[test]
fn separate_vec_add_crate_has_typed_mir_and_concrete_device_instances() {
    let report = support::checked(support::compile(
        "examples/vec-add/kernels/src/lib.rs",
        "metadata",
        true,
    ));
    assert!(report.contains("kernel: vec_add parameters=4"), "{report}");
    assert!(report.contains("read_buffer<f32>"), "{report}");
    assert!(report.contains("write_buffer<f32>"), "{report}");
    assert!(
        report.contains("ReadBuffer::<f32>::load_unchecked"),
        "{report}"
    );
    assert!(
        report.contains("WriteBuffer::<f32>::store_unchecked"),
        "{report}"
    );
    for builtin in ["thread_idx", "block_idx", "block_dim"] {
        assert!(report.contains(&format!("builtin: {builtin}")), "{report}");
    }
    assert!(report.contains("asserts=2"), "{report}");
}

#[test]
fn concrete_helpers_and_core_trait_implementations_are_resolved() {
    let report = support::checked(support::fixture("generics"));
    for instance in [
        "sum::<f32>",
        "sum::<u32>",
        "bias::<3>",
        "<f32 as core::ops::Add>::add",
        "<u32 as core::ops::Add>::add",
    ] {
        assert!(report.contains(instance), "missing {instance}:\n{report}");
    }
    assert!(
        report.contains("asserts=1"),
        "overflow assertion was not retained:\n{report}"
    );
    assert!(report.contains("builtin: grid_dim"), "{report}");
}

#[test]
fn rust_type_errors_are_reported() {
    support::rejected(support::fixture("type_error"), "mismatched types");
}

#[test]
fn rust_borrow_errors_are_reported() {
    support::rejected(support::fixture("borrow_error"), "cannot borrow");
}

#[test]
fn unsupported_kernel_signatures_are_reported() {
    for (fixture, error) in [
        ("safe_kernel", "kernel entrypoints must be unsafe"),
        (
            "generic_kernel",
            "kernel entrypoints must not have generic parameters",
        ),
        ("invalid_entry", "#[kernel] requires a free function"),
        ("bool_parameter", "unsupported kernel parameter type: bool"),
        ("slice_parameter", "unsupported kernel parameter type"),
    ] {
        support::rejected(support::fixture(fixture), error);
    }
}

#[test]
fn unsupported_reachable_calls_are_reported() {
    for (fixture, error) in [
        (
            "foreign_call",
            "foreign ABI calls are not supported in Metal kernels",
        ),
        (
            "indirect_call",
            "indirect calls are not supported in Metal kernels",
        ),
        ("recursion", "recursion is not supported in Metal kernels"),
        (
            "closure",
            "closures and callable shims are not supported in Metal kernels",
        ),
        (
            "destructor",
            "destructors are not supported in Metal kernels",
        ),
    ] {
        support::rejected(support::fixture(fixture), error);
    }
}

#[test]
fn device_extern_operations_do_not_require_mir_bodies() {
    let report = support::checked(support::fixture("intrinsic_declarations"));
    for operation in [
        "thread_idx",
        "block_idx",
        "block_dim",
        "grid_dim",
        "float_math",
        "float_conversion",
        "threadgroup_barrier",
        "simd_lane",
        "simd_size",
        "simd_group",
        "simd_count",
    ] {
        assert!(
            report.contains(&format!("builtin: {operation} mir=unavailable")),
            "{operation}:\n{report}"
        );
    }
    for operation in [
        "buffer_load",
        "buffer_store",
        "threadgroup_alloc",
        "atomic_add",
        "simd_sum",
        "simd_shuffle",
    ] {
        assert!(
            report.contains(&format!("builtin: {operation}")),
            "{report}"
        );
    }
}

#[test]
fn foreign_rust_aliases_cannot_claim_device_intrinsic_symbols() {
    support::rejected(
        support::fixture("foreign_rust_call"),
        "foreign ABI calls are not supported in Metal kernels",
    );
}

#[test]
fn device_declarations_require_the_exact_intrinsic_signature() {
    support::rejected(
        support::invalid_device("malformed_device", "malformed_device_client", &[]),
        "invalid device intrinsic signature",
    );
}

#[test]
fn shared_adapters_require_matching_type_and_length_parameters() {
    for variant in ["reversed", "fixed_length", "fixed_element"] {
        support::rejected(
            support::invalid_device(
                "malformed_shared",
                "malformed_shared_client",
                &["--cfg", variant],
            ),
            "invalid device intrinsic signature",
        );
    }
}

#[test]
fn generic_drops_are_checked_after_monomorphization() {
    let report = support::checked(support::fixture("generic_drop"));
    assert!(report.contains("consume::<u32>"), "{report}");
    support::rejected(
        support::fixture("generic_destructor"),
        "destructors are not supported in Metal kernels",
    );
}

#[test]
fn concrete_associated_types_have_the_same_parameter_layout() {
    let report = support::checked(support::fixture("associated_types"));
    assert!(report.contains("scalar<u32> size=4 align=4"), "{report}");
    assert!(report.contains("read_buffer<f32>"), "{report}");
}

#[test]
fn compile_time_closures_delegate_to_rustc_abi_queries() {
    let report = support::checked(support::fixture("const_closure"));
    assert!(report.contains("kernel: constant"), "{report}");
}

#[test]
fn distinct_generic_instances_form_a_finite_call_graph() {
    let report = support::checked(support::fixture("finite_generics"));
    for instance in ["forward::<u32>", "forward::<f32>"] {
        assert!(report.contains(instance), "{report}");
    }
}

#[test]
fn expanding_generic_instances_hit_a_bounded_diagnostic() {
    support::rejected(
        support::fixture("expanding_generics"),
        "device instance depth exceeds the rustc recursion limit",
    );
}
