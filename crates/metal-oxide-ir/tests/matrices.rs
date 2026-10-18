mod support;

use metal_oxide_ir::*;
use support::{assign, source, uint};

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

/// A kernel of one block over the locals `[unit, parameters.., locals..]`.
fn kernel(parameters: Vec<Type>, locals: Vec<Type>, statements: Vec<Statement>) -> Module {
    let count = parameters.len();
    Module {
        types: TypeTable::default(),
        functions: vec![Function {
            name: "kernel".into(),
            kernel: true,
            required_block: Some([64, 1, 1]),
            requires: Vec::new(),
            parameters: count,
            locals: [vec![Type::Unit], parameters, locals].concat(),
            blocks: vec![Block {
                statements,
                terminator: Terminator::Return,
                source: source(),
            }],
            source: source(),
        }],
    }
}

/// Loads an F16 block from `from`, accumulates its square in f32 and stores that to `to`.
fn product(from: Type, to: Type) -> Module {
    kernel(
        vec![from, to],
        vec![Type::Matrix(Scalar::F16), Type::Matrix(Scalar::F32)],
        vec![
            assign(
                3,
                matrix(
                    MatrixOp::Load { transposed: true },
                    vec![Operand::local(1), uint(0), uint(0), uint(8)],
                ),
            ),
            assign(
                4,
                matrix(MatrixOp::Splat, vec![Operand::Constant(Constant::F32(0))]),
            ),
            assign(
                4,
                matrix(
                    MatrixOp::MultiplyAccumulate,
                    vec![Operand::local(3), Operand::local(3), Operand::local(4)],
                ),
            ),
            assign(
                0,
                matrix(
                    MatrixOp::Store { transposed: false },
                    vec![
                        Operand::local(2),
                        uint(0),
                        uint(0),
                        uint(8),
                        Operand::local(4),
                    ],
                ),
            ),
        ],
    )
}

#[test]
fn matrix_operations_type_loads_products_and_stores() {
    let module = product(
        buffer(Scalar::F16, Access::Read),
        buffer(Scalar::F32, Access::Write),
    );
    let analyses = validate(&module).unwrap();
    assert!(analyses[0].cooperative && analyses[0].effects);
}

#[test]
fn matrix_operations_reject_mismatched_buffers_and_elements() {
    for (from, to) in [
        (
            buffer(Scalar::F16, Access::Write),
            buffer(Scalar::F32, Access::Write),
        ),
        (
            buffer(Scalar::F16, Access::Read),
            buffer(Scalar::F32, Access::Read),
        ),
        (
            buffer(Scalar::F16, Access::Read),
            buffer(Scalar::F16, Access::Write),
        ),
        (
            buffer(Scalar::U16, Access::Read),
            buffer(Scalar::F32, Access::Write),
        ),
    ] {
        let error = validate(&product(from, to)).unwrap_err();
        assert!(
            error
                .message
                .contains("invalid SIMD-group matrix operation"),
            "{error}"
        );
    }
}

#[test]
fn matrices_live_in_arrays_but_never_cross_the_abi() {
    let mut types = TypeTable::default();
    let array = types.intern(Aggregate::Array {
        element: Type::Matrix(Scalar::F32),
        length: 4,
    });
    assert!(types.is_owned(array));
    assert!(!types.is_abi_value(array));
    assert!(!types.is_abi_value(Type::Matrix(Scalar::F32)));
    let mut module = kernel(vec![], vec![array], vec![]);
    module.types = types;
    validate(&module).unwrap();
    let parameter = kernel(vec![Type::Matrix(Scalar::F32)], vec![], vec![]);
    let error = validate(&parameter).unwrap_err();
    assert!(
        error.message.contains("unsupported kernel parameter type"),
        "{error}"
    );
    let local = kernel(vec![], vec![Type::Matrix(Scalar::U32)], vec![]);
    let error = validate(&local).unwrap_err();
    assert!(error.message.contains("unsupported local type"), "{error}");
}

fn read_write(element: Scalar, address_space: AddressSpace) -> Type {
    Type::Buffer {
        element: Element::Scalar(element),
        access: Access::ReadWrite,
        address_space,
    }
}

/// The type of `expression` over the locals `[unit, locals..]`.
fn typed(locals: Vec<Type>, expression: Expression) -> Result<Type, Error> {
    let module = kernel(vec![], locals, vec![]);
    expression_type(&module, &module.functions[0], &expression, &source())
}

fn assert_invalid(result: Result<Type, Error>) {
    let error = result.unwrap_err();
    assert!(
        error
            .message
            .contains("invalid SIMD-group matrix operation"),
        "{error}"
    );
}

/// `[buffer, row, column, stride, matrix]` over the locals `[unit, buffer, matrix]`.
fn block_operands() -> Vec<Operand> {
    let positions = [uint(0), uint(0), uint(8)];
    [
        vec![Operand::local(1)],
        positions.to_vec(),
        vec![Operand::local(2)],
    ]
    .concat()
}

fn load(operands: &[Operand]) -> Expression {
    matrix(MatrixOp::Load { transposed: false }, operands[..4].to_vec())
}

fn store(operands: &[Operand]) -> Expression {
    matrix(MatrixOp::Store { transposed: false }, operands.to_vec())
}

#[test]
fn read_write_buffers_accept_matrix_loads_and_stores() {
    for space in [AddressSpace::Device, AddressSpace::Threadgroup] {
        for element in [Scalar::F32, Scalar::F16, Scalar::BF16] {
            let locals = vec![read_write(element, space), Type::Matrix(element)];
            let operands = block_operands();
            let loaded = typed(locals.clone(), load(&operands));
            assert_eq!(loaded, Ok(Type::Matrix(element)), "{space:?} {element:?}");
            let stored = typed(locals, store(&operands));
            assert_eq!(stored, Ok(Type::Unit), "{space:?} {element:?}");
        }
    }
}

#[test]
fn atomic_buffers_reject_matrix_loads_and_stores() {
    let locals = vec![
        buffer(Scalar::F32, Access::Atomic),
        Type::Matrix(Scalar::F32),
    ];
    let operands = block_operands();
    assert_invalid(typed(locals.clone(), load(&operands)));
    assert_invalid(typed(locals, store(&operands)));
}

#[test]
fn matrix_loads_and_stores_need_u32_positions() {
    let locals = vec![
        read_write(Scalar::F32, AddressSpace::Device),
        Type::Matrix(Scalar::F32),
    ];
    for position in 1..=3 {
        for value in [Constant::Usize(0), Constant::I32(0), Constant::F32(0)] {
            let mut operands = block_operands();
            operands[position] = Operand::Constant(value);
            assert_invalid(typed(locals.clone(), load(&operands)));
            assert_invalid(typed(locals.clone(), store(&operands)));
        }
    }
}

#[test]
fn a_splat_takes_a_matrix_element_value() {
    let splat = |value: Constant| matrix(MatrixOp::Splat, vec![Operand::Constant(value)]);
    for (value, element) in [
        (Constant::F32(0), Scalar::F32),
        (Constant::F16(0), Scalar::F16),
        (Constant::BF16(0), Scalar::BF16),
    ] {
        assert_eq!(typed(vec![], splat(value)), Ok(Type::Matrix(element)));
    }
    for value in [Constant::U32(0), Constant::I32(0), Constant::Bool(false)] {
        assert_invalid(typed(vec![], splat(value)));
    }
}

#[test]
fn multiply_accumulate_takes_three_matrices() {
    let locals = vec![
        Type::Matrix(Scalar::F16),
        Type::Matrix(Scalar::F32),
        Type::Scalar(Scalar::F32),
    ];
    let multiply = |operands: [usize; 3]| {
        let operands = operands.map(Operand::local).to_vec();
        typed(
            locals.clone(),
            matrix(MatrixOp::MultiplyAccumulate, operands),
        )
    };
    assert_eq!(multiply([1, 1, 2]), Ok(Type::Matrix(Scalar::F32)));
    for operands in [[3, 1, 2], [1, 3, 2], [1, 1, 3]] {
        assert_invalid(multiply(operands));
    }
}

#[test]
fn matrix_operations_take_exactly_their_operands() {
    let locals = vec![
        read_write(Scalar::F32, AddressSpace::Device),
        Type::Matrix(Scalar::F32),
    ];
    let operands = block_operands();
    let float = Operand::Constant(Constant::F32(0));
    for (op, valid) in [
        (MatrixOp::Splat, vec![float.clone()]),
        (MatrixOp::MultiplyAccumulate, vec![Operand::local(2); 3]),
        (MatrixOp::Load { transposed: true }, operands[..4].to_vec()),
        (MatrixOp::Store { transposed: false }, operands),
    ] {
        assert_eq!(valid.len(), op.arity());
        typed(locals.clone(), matrix(op, valid.clone())).unwrap();
        assert_invalid(typed(locals.clone(), matrix(op, valid[1..].to_vec())));
        let more = [valid, vec![float.clone()]].concat();
        assert_invalid(typed(locals.clone(), matrix(op, more)));
    }
}

#[test]
fn the_operand_count_is_checked_before_the_operands_are_typed() {
    let malformed = Operand::AggregateConstant {
        ty: Type::Unit,
        fields: vec![],
    };
    let splat = matrix(MatrixOp::Splat, vec![malformed.clone(), malformed]);
    assert_invalid(typed(vec![], splat));
}

#[test]
fn matrices_of_other_elements_fail_type_validation_in_aggregates() {
    let validated = |element: Scalar| {
        let mut types = TypeTable::default();
        let pair = types.intern(Aggregate::Array {
            element: Type::Matrix(element),
            length: 2,
        });
        let mut module = kernel(vec![], vec![pair], vec![]);
        module.types = types;
        validate(&module)
    };
    for element in [Scalar::F32, Scalar::F16, Scalar::BF16] {
        validated(element).unwrap();
    }
    let error = validated(Scalar::U32).unwrap_err();
    assert!(
        error.message.contains("requires value components"),
        "{error}"
    );
}
