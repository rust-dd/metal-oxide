#[allow(dead_code)]
mod support;

use metal_oxide_ir::*;
use support::{block, source, uint};

fn matrix(op: MatrixOp, arguments: Vec<Operand>) -> Expression {
    Expression::Matrix { op, arguments }
}

fn buffer(element: Scalar, access: Access) -> Type {
    Type::Buffer {
        element: Element::Scalar(element),
        access,
        address_space: AddressSpace::Device,
    }
}

/// Squares an `element` block, loaded from row 8 or splat when `load` is unset, into an f32
/// accumulator and stores it transposed at column 8.
fn product(element: Scalar, load: bool) -> Module {
    let mut types = TypeTable::default();
    let pair = types.intern(Aggregate::Array {
        element: Type::Matrix(Scalar::F32),
        length: 2,
    });
    let block_matrix = if load {
        matrix(
            MatrixOp::Load { transposed: false },
            vec![Operand::local(1), uint(8), uint(0), uint(16)],
        )
    } else {
        assert_eq!(element, Scalar::F32);
        matrix(MatrixOp::Splat, vec![Operand::Constant(Constant::F32(0))])
    };
    let statements = vec![
        (3, block_matrix),
        (
            4,
            matrix(MatrixOp::Splat, vec![Operand::Constant(Constant::F32(0))]),
        ),
        (
            4,
            matrix(
                MatrixOp::MultiplyAccumulate,
                vec![Operand::local(3), Operand::local(3), Operand::local(4)],
            ),
        ),
        (
            5,
            Expression::Aggregate {
                ty: 0,
                fields: vec![Operand::local(4), Operand::local(4)],
            },
        ),
        (
            0,
            matrix(
                MatrixOp::Store { transposed: true },
                vec![
                    Operand::local(2),
                    uint(0),
                    uint(8),
                    uint(16),
                    Operand::local(4),
                ],
            ),
        ),
    ];
    Module {
        types,
        functions: vec![Function {
            name: "product".into(),
            kernel: true,
            required_block: Some([32, 1, 1]),
            requires: Vec::new(),
            parameters: 2,
            locals: vec![
                Type::Unit,
                buffer(element, Access::Read),
                buffer(Scalar::F32, Access::Write),
                Type::Matrix(element),
                Type::Matrix(Scalar::F32),
                pair,
            ],
            blocks: vec![block(statements, Terminator::Return)],
            source: source(),
        }],
    }
}

#[test]
fn matrix_operations_emit_simdgroup_matrices_through_value_helpers() {
    let module = product(Scalar::F16, true);
    let codegen = metal_oxide_codegen::Codegen::new(&module).unwrap();
    let msl = codegen.emit().unwrap();
    for expected in [
        "simdgroup_matrix<half, 8, 8>",
        "metal_oxide_matrix_load<half>(",
        "ulong(16u), ulong2(0u, 8u), false)",
        "make_filled_simdgroup_matrix<float, 8, 8>(",
        "metal_oxide_matrix_multiply_accumulate(",
        "simdgroup_store(",
        "ulong(16u), ulong2(8u, 0u), true)",
        "static_assert(sizeof(metal_oxide_aggregate_0) == 512 && alignof(metal_oxide_aggregate_0) == 256",
    ] {
        assert!(msl.contains(expected), "missing {expected:?} in\n{msl}");
    }
    assert_eq!(msl.matches("metal_oxide_matrix_load(P source").count(), 1);
    assert_eq!(
        codegen.abi().unwrap().required_features,
        ["simdgroup_matrix"]
    );
}

#[test]
fn bfloat_matrices_also_require_bfloat16_at_the_msl_baseline() {
    let module = product(Scalar::BF16, true);
    let codegen = metal_oxide_codegen::Codegen::new(&module).unwrap();
    assert!(
        codegen
            .emit()
            .unwrap()
            .contains("simdgroup_matrix<bfloat, 8, 8>")
    );
    let abi = codegen.abi().unwrap();
    assert_eq!(abi.required_features, ["simdgroup_matrix", "bfloat16"]);
    assert_eq!(
        abi.target_flags(),
        ["-std=metal3.2", "-mmacosx-version-min=15.0"]
    );
}

#[test]
fn modules_without_matrix_loads_or_products_omit_the_load_helper() {
    let mut module = product(Scalar::F32, false);
    module.functions[0].blocks[0]
        .statements
        .retain(|statement| {
            !matches!(
                statement.value,
                Expression::Matrix {
                    op: MatrixOp::MultiplyAccumulate,
                    ..
                }
            )
        });
    let msl = metal_oxide_codegen::Codegen::new(&module)
        .unwrap()
        .emit()
        .unwrap();
    assert!(!msl.contains("template <typename T, typename P>"), "{msl}");
    assert!(msl.contains("make_filled_simdgroup_matrix<float, 8, 8>("));
}

/// A kernel that holds a pair of `element` matrices.
fn pair_of(element: Scalar) -> Module {
    let mut types = TypeTable::default();
    types.intern(Aggregate::Array {
        element: Type::Matrix(element),
        length: 2,
    });
    Module {
        types,
        functions: vec![Function {
            name: "pair".into(),
            kernel: true,
            required_block: Some([32, 1, 1]),
            requires: Vec::new(),
            parameters: 0,
            locals: vec![Type::Unit, Type::Aggregate(0)],
            blocks: vec![block(vec![], Terminator::Return)],
            source: source(),
        }],
    }
}

#[test]
fn arrays_of_half_and_bfloat_matrices_lay_out_by_their_element_size() {
    for element in [Scalar::F16, Scalar::BF16] {
        let msl = metal_oxide_codegen::Codegen::new(&pair_of(element))
            .unwrap()
            .emit()
            .unwrap();
        let layout =
            "sizeof(metal_oxide_aggregate_0) == 256 && alignof(metal_oxide_aggregate_0) == 128";
        assert!(
            msl.contains(layout),
            "missing {layout:?} for {element:?} in\n{msl}"
        );
    }
}
