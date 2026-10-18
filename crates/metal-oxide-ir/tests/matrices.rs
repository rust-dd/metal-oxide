use metal_oxide_ir::*;

fn source() -> SourceLocation {
    SourceLocation {
        file: "kernel.rs".into(),
        line: 3,
        column: 5,
    }
}

fn assign(local: usize, value: Expression) -> Statement {
    Statement {
        destination: Place::local(local),
        value,
        source: source(),
    }
}

fn uint(value: u32) -> Operand {
    Operand::Constant(Constant::U32(value))
}

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
