use super::support;
use metal_oxide_artifact::{Layout, LayoutKind, ParameterType, Scalar};

#[test]
fn unknown_i64_overflow_and_division_are_checked_at_runtime() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/i64_checked.rs",
        &[],
    );
    support::checked(output);
    let zero = support::check(&directory, serde_json::json!("division_by_zero"), 6);
    let division = support::check(&directory, serde_json::json!({"overflow": "div"}), 6);
    let multiply = support::check(&directory, serde_json::json!({"overflow": "mul"}), 7);
    let add = support::check(&directory, serde_json::json!({"overflow": "add"}), 7);
    let output = support::execute_msl(
        &directory,
        r#"
        long minimum = -0x7fffffffffffffffl - 1;
        long cases[7][2] = {{3, -4}, {-3, -4}, {1l << 32, 1l << 32}, {0x7fffffffffffffffl, 1}, {7, 0}, {minimum, -1}, {minimum, 2}};
        for (auto& c : cases) {
            long out[2] = {5, 5};
            atomic_store_explicit(metal_oxide_failure, 0u, memory_order_relaxed);
            checked(out, c[0], c[1], {}, {}, {1, 1, 1}, {1, 1, 1}, metal_oxide_gate, metal_oxide_failure);
            std::cout << out[0] << ',' << out[1] << ',' << atomic_load_explicit(metal_oxide_failure, memory_order_relaxed) << ';';
        }
    "#,
    );
    assert_eq!(
        output,
        format!(
            "0,-11,0;0,13,0;1,5,{};{},5,{};5,5,{};5,5,{};{},5,{};",
            multiply + 2,
            i64::MAX,
            add + 2,
            zero + 2,
            division + 2,
            i64::MIN / 2,
            multiply + 2,
        )
    );
}

#[test]
fn i64_records_and_arrays_cross_the_abi_as_aligned_eight_byte_scalars() {
    let (output, directory) = support::emit(
        "crates/metal-oxide-compiler/tests/fixtures/i64.rs",
        &["-C", "overflow-checks=off"],
    );
    support::checked(output);
    let abi = metal_oxide_artifact::Abi::from_json(
        &std::fs::read_to_string(directory.join("abi.json")).unwrap(),
    )
    .unwrap();
    let kernel = abi
        .kernels
        .iter()
        .find(|kernel| kernel.name == "signed_wide_ops")
        .unwrap();
    let ParameterType::Buffer {
        element, stride, ..
    } = &kernel.parameters[0].ty
    else {
        panic!("{:?}", kernel.parameters[0].ty)
    };
    let LayoutKind::Tuple { fields } = &element.kind else {
        panic!("{element:?}")
    };
    assert_eq!(
        fields
            .iter()
            .map(|field| (field.offset, field.layout.scalar_type()))
            .collect::<Vec<_>>(),
        [
            (0, Some(Scalar::I32)),
            (8, Some(Scalar::I64)),
            (16, Some(Scalar::I64)),
            (24, Some(Scalar::U32)),
        ]
    );
    assert_eq!((element.size, element.alignment, *stride), (32, 8, 32));
    let ParameterType::Buffer { element, .. } = &kernel.parameters[1].ty else {
        panic!("{:?}", kernel.parameters[1].ty)
    };
    assert_eq!(
        *element,
        Layout::array(Layout::scalar(Scalar::I64), 37).unwrap()
    );
    let msl = std::fs::read_to_string(directory.join("kernels.metal")).unwrap();
    assert!(
        msl.contains("metal_oxide_checked_i64 metal_oxide_mul_checked_i64(long a, long b)"),
        "{msl}"
    );
}
