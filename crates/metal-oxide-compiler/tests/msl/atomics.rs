use super::support;

#[test]
fn float_boolean_tuples_construct_return_and_destructure() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/float_boolean_tuple.rs",
        &[],
    );
    support::checked(output);
    let actual = support::execute_msl(
        &directory,
        r#"
        for (float value : {-7.5f, 0.0f, 2.25f}) {
            float values[1] = {}; uint flags[1] = {};
            float_boolean_tuple(values, flags, value, {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate);
            std::cout << values[0] << ',' << flags[0] << ';';
        }
        "#,
    );
    assert_eq!(actual, "-15,0;0,0;4.5,1;");
}

#[test]
fn numeric_atomics_record_capabilities_and_exact_operations() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/atomic_numeric.rs",
        &[],
    );
    support::checked(output);
    let source = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    for operation in [
        "atomic_float",
        "atomic_ulong",
        "atomic_min_explicit",
        "atomic_max_explicit",
        "atomic_fetch_add_explicit",
        "atomic_fetch_sub_explicit",
    ] {
        assert!(source.contains(operation), "missing {operation}");
    }
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(abi.required_features, ["float32_atomics", "uint64_atomics"]);
    assert_eq!(abi.msl_version(), "3.2");
}

#[test]
fn ordered_atomics_require_msl_41_and_order_both_address_spaces() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/atomic_ordering.rs",
        &[],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        abi.required_features,
        ["int32_atomics", "atomic_acquire_release"]
    );
    assert_eq!(
        abi.target_flags(),
        ["-std=metal4.1", "-mmacosx-version-min=27.0"]
    );
    let source = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    for ordering in [
        "memory_order_acquire",
        "memory_order_release",
        "memory_order_acq_rel",
    ] {
        assert!(source.contains(ordering));
    }
    assert_eq!(
        source
            .matches("mem_flags::mem_device | mem_flags::mem_threadgroup")
            .count(),
        5
    );
    assert!(source.contains("memory_order_acq_rel, memory_order_acquire"));
    assert!(source.contains("memory_order_release, memory_order_relaxed"));
}

#[test]
fn unsupported_atomic_operations_and_orderings_are_diagnosed() {
    for (element, expression, message) in [
        (
            "u32",
            "{ let order = if cell.load(0, Relaxed) == 0 { Relaxed } else { Acquire }; cell.load(0, order) }",
            "atomic ordering must be a compile-time constant",
        ),
        (
            "u32",
            "cell.load(0, SeqCst)",
            "sequentially consistent atomics are unsupported",
        ),
        (
            "u32",
            "cell.load(0, Release)",
            "invalid atomic memory ordering",
        ),
        (
            "u32",
            "cell.store(0, 1, Acquire)",
            "invalid atomic memory ordering",
        ),
        (
            "u32",
            "cell.compare_exchange_weak(0, 1, 2, Relaxed, Acquire)",
            "invalid atomic memory ordering",
        ),
        (
            "u32",
            "cell.compare_exchange_weak(0, 1, 2, AcqRel, Release)",
            "invalid atomic memory ordering",
        ),
        (
            "f32",
            "cell.fetch_min(0, 1.0, Relaxed)",
            "unsupported atomic operation for this element and address space",
        ),
        (
            "u64",
            "cell.fetch_add(0, 1, Relaxed)",
            "unsupported atomic operation for this element and address space",
        ),
        (
            "u64",
            "cell.min(0, 1, Acquire)",
            "invalid atomic memory ordering",
        ),
    ] {
        let directory = support::root()
            .join("target/atomic-rejections")
            .join(std::process::id().to_string());
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("rejected.rs");
        std::fs::write(&source, format!("#![no_std]\nuse core::sync::atomic::Ordering::*;\nuse metal_oxide_device::{{kernel, AtomicBuffer}};\n#[kernel] pub unsafe fn rejected(cell: AtomicBuffer<{element}>) {{ unsafe {{ {expression}; }} }}")).unwrap();
        let (output, _) = support::emit(source.to_str().unwrap(), &[]);
        support::rejected(output, message);
    }
}
